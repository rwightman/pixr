""" Read task: transcribe all text on a document page (OCR-free pretraining objective), w/ CER / WER evaluation. """
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import chugr
from chugr.doc import SynthTextCfg

from pixr.framework import DeviceEnv, TaskCfg
from pixr.utils import text_error_rates
from .common import ImageTextTask, image_feature_info

_logger = logging.getLogger(__name__)


@dataclass
class ReadTaskCfg(TaskCfg):
    text_max_length: int = 1024  # max target tokens per page
    page_sampling: str = 'random'  # page sampling for multi-page docs during training ('random', 'first')
    render_dpi: int = 150  # pdf render dpi (before resize to model input)
    num_metrics_samples: int = 4  # train samples to decode (greedy) for train CER / WER
    metrics_max_new_tokens: int = 512
    eval_max_new_tokens: int = 1023
    eval_num_beams: int = 1  # beam search for eval generation, 1 == greedy
    eval_page_sampling: str = 'first'
    # synthetic text rendering (of the sample's own annotation text, target == rendered text, no bboxes used)
    synth_prob: float = 0.  # train, probability of a synthetic rendering replacing the page
    eval_synth_prob: float = 0.  # eval, 1.0 for held-out synthetic reading CER
    synth: SynthTextCfg = field(default_factory=SynthTextCfg)


class ReadTask(ImageTextTask):
    task_name = 'read'
    task_start_token = '<s_pretrain>'  # NOTE name kept for checkpoint token compatibility
    task_tokens = (task_start_token,)

    def __init__(self, cfg: ReadTaskCfg, device_env: DeviceEnv, checkpoint_path: str = ''):
        super().__init__(cfg, device_env, checkpoint_path=checkpoint_path)
        self.text_preprocess = chugr.create_text_preprocessor(
            self.tokenizer,
            max_length=cfg.text_max_length,
            task_start_token=self.task_start_token,
            prompt_end_token=self.task_start_token,
        )
        self.results = []

    def train_data_cfg(self):
        return chugr.DataTaskDocReadCfg(
            image_process_fn=self.image_preprocess_train,
            image_process_factory=self.image_preprocess_factory,
            text_process_fn=self.text_preprocess,
            image_input_feat=image_feature_info(self.input_info['image_fmt']),
            page_sampling=self.cfg.page_sampling,
            render_dpi=self.cfg.render_dpi,
            synth_prob=self.cfg.synth_prob,
            synth_cfg=self.cfg.synth,
            error_handler='log_and_continue',
        )

    def eval_data_cfg(self):
        return chugr.DataTaskDocReadCfg(
            image_process_fn=self.image_preprocess_eval,
            text_process_fn=None,
            text_target_feat=None,
            image_input_feat=image_feature_info(self.input_info['image_fmt']),
            page_sampling=self.cfg.eval_page_sampling,
            render_dpi=self.cfg.render_dpi,
            synth_prob=self.cfg.eval_synth_prob,
            synth_cfg=self.cfg.synth,
            error_handler='log_and_continue',
        )

    def _prompts(self, n):
        return [self.tokenizer.convert_tokens_to_ids([self.task_start_token])] * n

    def train_metrics(self, batch: Dict[str, Any], outputs: Dict[str, Any]):
        """ Greedy decode a few train samples, CER / WER vs the targets (both capped at metrics_max_new_tokens). """
        n = min(self.cfg.num_metrics_samples, batch['text_input'].shape[0])
        image_input = batch['image_input']
        if isinstance(image_input, dict):
            # NOTE patch_valid is None for batches w/o padding
            image_input = {k: v[:n] if v is not None else None for k, v in image_input.items()}
        else:
            image_input = image_input[:n]
        generated = self.generate(
            image_input, self._prompts(n), [self.tokenizer.eos_token_id], self.cfg.metrics_max_new_tokens)
        predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        targets = []
        for t in batch['text_input'][:n]:
            t = t[t != self.pad_token_id][1:]  # drop start token
            # NOTE not truncated to the generated length, an early EOS must count against the prediction
            targets.append(self.tokenizer.decode(t[:self.cfg.metrics_max_new_tokens], skip_special_tokens=True))
        rates = text_error_rates(predictions, targets)
        _logger.info(f'Sample prediction: {predictions[0][:300]!r}')
        _logger.info(f'Sample target:     {targets[0][:300]!r}')
        return {'train_cer': rates['cer'], 'train_wer': rates['wer']}, {'prediction': predictions[0], 'target': targets[0]}

    def _prepare_reference(self, text: str) -> Tuple[str, bool]:
        """ Reference text for eval metrics, returns (text, truncated).

        * Every reference takes the same tokenizer round trip as the predictions (decoded token ids), so both
          sides share its text normalization (whitespace, characters the tokenizer can't represent).
        * Capped at the eval generation budget (eval_max_new_tokens). Pages w/ longer transcriptions can't be
          fully predicted, w/o the cap their CER is inflated by the unreachable tail.
        """
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        max_tokens = self.cfg.eval_max_new_tokens
        return self.tokenizer.decode(ids[:max_tokens], skip_special_tokens=True), len(ids) > max_tokens

    def evaluate(self, loader, use_ema: bool = False) -> Dict[str, Any]:
        predictions, references = [], []
        num_truncated = 0
        for batch in loader:
            texts = batch['text_input']
            generated = self.generate(
                batch['image_input'], self._prompts(len(texts)), [self.tokenizer.eos_token_id],
                self.cfg.eval_max_new_tokens, use_ema=use_ema, num_beams=self.cfg.eval_num_beams)
            predictions.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
            for t in texts:
                ref, truncated = self._prepare_reference(t[0] if isinstance(t, (list, tuple)) else t)
                references.append(ref)
                num_truncated += truncated
        if self.device_env.world_size > 1:
            predictions = [p for r in self.device_env.all_gather_object(predictions) for p in r]
            references = [p for r in self.device_env.all_gather_object(references) for p in r]
            num_truncated = sum(self.device_env.all_gather_object(num_truncated))
        self.results = [{'prediction': p, 'reference': r} for p, r in zip(predictions, references)]
        metrics = text_error_rates(predictions, references)
        metrics['num_samples'] = len(predictions)
        metrics['truncated_refs'] = num_truncated / max(len(references), 1)  # fraction of capped references
        return metrics

    def save_results(self, output_dir: str, name: str = 'read'):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{name}_results.json'), 'w') as f:
            json.dump(self.results, f, indent=1)
