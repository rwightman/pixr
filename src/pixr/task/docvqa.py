""" DocVQA task (Donut style prompt / answer format), fine-tuning w/ answer only loss and ANLS evaluation.

'<s_docvqa><s_question>{question}</s_question><s_answer>{answer}</s_answer><eos>'
"""
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

import chugr

from pixr.framework import DeviceEnv, TaskCfg
from pixr.utils import anls_score
from .common import ImageTextTask, image_feature_info

_logger = logging.getLogger(__name__)

DOCVQA_START_TOKEN = '<s_docvqa>'
DOCVQA_TOKENS = (DOCVQA_START_TOKEN, '<s_question>', '</s_question>', '<s_answer>', '</s_answer>')


@dataclass
class DocVqaTaskCfg(TaskCfg):
    text_max_length: int = 128
    eval_max_new_tokens: int = 64


class DocVqaTask(ImageTextTask):
    task_name = 'docvqa'
    task_tokens = DOCVQA_TOKENS
    prompt_end_token = '<s_answer>'

    def __init__(self, cfg: DocVqaTaskCfg, device_env: DeviceEnv, checkpoint_path: str = ''):
        super().__init__(cfg, device_env, checkpoint_path=checkpoint_path)
        self.text_preprocess = chugr.create_text_preprocessor(
            self.tokenizer,
            max_length=cfg.text_max_length,
            task_start_token=DOCVQA_START_TOKEN,
            prompt_end_token=self.prompt_end_token,  # loss on the answer only
        )
        self.stop_token_ids = [self.tokenizer.convert_tokens_to_ids('</s_answer>'), self.tokenizer.eos_token_id]
        self.results = []

    def _data_cfg(self, is_training: bool):
        return chugr.DataTaskDocVqaCfg(
            image_process_fn=self.image_preprocess_train if is_training else self.image_preprocess_eval,
            image_process_factory=self.image_preprocess_factory if is_training else None,
            text_process_fn=self.text_preprocess if is_training else None,
            image_input_feat=image_feature_info(self.input_info['image_fmt']),
            text_target_feat=chugr.FeatureInfo('text_target', input_key=None) if is_training else None,
            error_handler='log_and_continue' if is_training else 'reraise_exception',
        )

    def train_data_cfg(self):
        return self._data_cfg(is_training=True)

    def eval_data_cfg(self):
        # eval text input is the prompt string '<s_question>...</s_question><s_answer>'
        return self._data_cfg(is_training=False)

    def evaluate(self, loader, use_ema: bool = False) -> Dict[str, Any]:
        results = []
        for batch in loader:
            prompts = [self.tokenizer(DOCVQA_START_TOKEN + p, add_special_tokens=False).input_ids
                       for p in batch['text_input']]
            generated = self.generate(
                batch['image_input'], prompts, self.stop_token_ids, self.cfg.eval_max_new_tokens, use_ema=use_ema)
            answers = [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in generated]
            for answer, parse in zip(answers, batch['_parse']):
                gt = parse.get('answers') or []
                results.append({
                    'questionId': parse.get('question_id'),
                    'question': parse.get('question'),
                    'answer': answer,
                    'gt_answers': list(gt),
                    'anls': anls_score(answer, gt) if gt else None,
                })
        if self.device_env.world_size > 1:
            merged = {}
            for i, r in enumerate([r for rank in self.device_env.all_gather_object(results) for r in rank]):
                # de-dupe samples padded by the (HF dataset) distributed sampler, samples w/o ids are kept
                merged.setdefault(r['questionId'] if r['questionId'] is not None else ('_no_id', i), r)
            results = list(merged.values())
        self.results = results
        scored = [r['anls'] for r in results if r['anls'] is not None]
        return {'anls': sum(scored) / max(len(scored), 1), 'num_samples': len(results)}

    def save_results(self, output_dir: str, name: str = 'docvqa'):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{name}_results.json'), 'w') as f:
            json.dump(self.results, f, indent=1)
        # DocVQA eval server submission format
        with open(os.path.join(output_dir, f'{name}_submission.json'), 'w') as f:
            json.dump([{'questionId': r['questionId'], 'answer': r['answer']} for r in self.results], f)
