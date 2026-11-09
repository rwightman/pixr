import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

import chugr

from pixr.data import create_image_preprocessor
from pixr.framework import DeviceEnv, Monitor, TaskEval, TaskEvalCfg
from pixr.generation import generate_greedy
from pixr.utils import anls_score
from .common import init_model_and_tokenizer, image_feature_info, to_device

_logger = logging.getLogger(__name__)

DOCVQA_START_TOKEN = '<s_docvqa>'
DOCVQA_TOKENS = [DOCVQA_START_TOKEN, '<s_question>', '</s_question>', '<s_answer>', '</s_answer>']


def docvqa_task_cfg(
        image_process_fn,
        image_fmt: str,
        text_process_fn=None,
        is_training: bool = True,
):
    """ chugr DocVQA pipeline cfg. Samples: '<s_question>{q}</s_question><s_answer>[{a}</s_answer>]'.

    For eval (no text_process_fn / target) the text input is the prompt string up to '<s_answer>'.
    """
    return chugr.DataTaskDocVqaCfg(
        image_process_fn=image_process_fn,
        text_process_fn=text_process_fn,
        image_input_feat=image_feature_info(image_fmt),
        text_target_feat=chugr.FeatureInfo('text_target', input_key=None) if is_training else None,
        error_handler='log_and_continue' if is_training else 'reraise_exception',
    )


class DocVqaEvaluator:
    """ Greedy decode answers for DocVQA samples, accumulate predictions and ANLS. """

    def __init__(self, tokenizer, device, autocast, max_new_tokens: int = 64):
        self.tokenizer = tokenizer
        self.device = device
        self.autocast = autocast
        self.max_new_tokens = max_new_tokens
        self.stop_token_ids = [
            tokenizer.convert_tokens_to_ids('</s_answer>'),
            tokenizer.eos_token_id,
        ]
        self.results = []

    def reset(self):
        self.results = []

    @torch.no_grad()
    def step(self, model, batch):
        image_input = to_device(batch['image_input'], self.device)
        prompts = [DOCVQA_START_TOKEN + p for p in batch['text_input']]
        prompt_ids = [self.tokenizer(p, add_special_tokens=False).input_ids for p in prompts]
        with self.autocast():
            generated = generate_greedy(
                model,
                image_input,
                prompt_ids=prompt_ids,
                stop_token_ids=self.stop_token_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=self.max_new_tokens,
            )
        answers = [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in generated]
        for answer, parse in zip(answers, batch['_parse']):
            gt = parse.get('answers') or []
            self.results.append({
                'questionId': parse.get('question_id'),
                'question': parse.get('question'),
                'answer': answer,
                'gt_answers': list(gt),
                'anls': anls_score(answer, gt) if gt else None,
            })

    def gather(self, device_env: DeviceEnv):
        if device_env.world_size > 1:
            gathered = device_env.all_gather_object(self.results)
            results = {}
            for r in [r for rank_results in gathered for r in rank_results]:
                results.setdefault(r['questionId'], r)  # de-dupe samples padded by the distributed sampler
            self.results = list(results.values())
        return self.results

    def metrics(self) -> Dict[str, float]:
        scored = [r['anls'] for r in self.results if r['anls'] is not None]
        return {'anls': sum(scored) / max(len(scored), 1), 'num_samples': len(self.results)}


def evaluate_docvqa(model, loader, evaluator: DocVqaEvaluator, device_env: DeviceEnv, log_frequency: int = 20):
    was_training = model.training
    model.eval()
    evaluator.reset()
    for batch_idx, batch in enumerate(loader):
        evaluator.step(model, batch)
        if log_frequency and batch_idx % log_frequency == 0:
            _logger.info(f'DocVQA eval batch {batch_idx}/{getattr(loader, "num_batches", "?")}')
    evaluator.gather(device_env)
    if was_training:
        model.train()
    return evaluator.metrics()


@dataclass
class TaskCrullerEvalDocVqaCfg(TaskEvalCfg):
    max_new_tokens: int = 64


class TaskCrullerEvalDocVqa(TaskEval):
    """ Evaluate a DocVQA fine-tuned checkpoint, ANLS on labelled splits + predictions json for submission. """
    task_name = 'cruller_eval_docvqa'

    def __init__(
            self,
            cfg: TaskCrullerEvalDocVqaCfg,
            device_env: DeviceEnv,
            monitor: Monitor = None,
            checkpoint_path: str = '',
    ):
        super().__init__(cfg=cfg, device_env=device_env, monitor=monitor)
        assert checkpoint_path, 'A checkpoint is required for evaluation.'
        self.model, self.tokenizer = init_model_and_tokenizer(
            cfg.model,
            cfg.tokenizer,
            task_tokens=DOCVQA_TOKENS,
            checkpoint_path=checkpoint_path,
        )
        self.input_info = self.model.image_encoder.input_info
        self.image_preprocess_eval = create_image_preprocessor(
            self.input_info,
            is_training=False,
            transform_type=cfg.image_transforms,
        )
        self.evaluator = DocVqaEvaluator(
            self.tokenizer,
            device=device_env.device,
            autocast=self.autocast,
            max_new_tokens=cfg.max_new_tokens,
        )

    def eval_data_cfg(self):
        return docvqa_task_cfg(self.image_preprocess_eval, self.input_info['image_fmt'], is_training=False)

    def setup(self):
        self.model.to(self.device_env.device)
        self.model.eval()

    def step(self, batch):
        self.evaluator.step(self.model, batch)

    def evaluate(self, loader):
        return evaluate_docvqa(self.model, loader, self.evaluator, self.device_env)

    def save_results(self, output_dir: str, name: str = 'docvqa'):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{name}_results.json'), 'w') as f:
            json.dump(self.evaluator.results, f, indent=1)
        # DocVQA submission format
        with open(os.path.join(output_dir, f'{name}_submission.json'), 'w') as f:
            json.dump([{'questionId': r['questionId'], 'answer': r['answer']} for r in self.evaluator.results], f)
