import json
import logging
import os
from dataclasses import dataclass

import torch

import chugr

from pixr.data import create_image_preprocessor
from pixr.framework import DeviceEnv, Monitor, TaskEval, TaskEvalCfg
from pixr.generation import generate_greedy
from pixr.utils import text_error_rates
from .common import init_model_and_tokenizer, image_feature_info, to_device
from .task_cruller_pretrain import TaskCrullerPretrain

_logger = logging.getLogger(__name__)


@dataclass
class TaskCrullerEvalOcrCfg(TaskEvalCfg):
    max_new_tokens: int = 1023
    page_sampling: str = 'first'
    render_dpi: int = 150


class TaskCrullerEvalOcr(TaskEval):
    """ Evaluate full page reading (pretrain objective) w/ corpus CER / WER. """
    task_name = 'cruller_eval_ocr'

    def __init__(
            self,
            cfg: TaskCrullerEvalOcrCfg,
            device_env: DeviceEnv,
            monitor: Monitor = None,
            checkpoint_path: str = '',
    ):
        super().__init__(cfg=cfg, device_env=device_env, monitor=monitor)
        assert checkpoint_path, 'A checkpoint is required for evaluation.'
        self.task_start_token = TaskCrullerPretrain.task_start_token
        self.model, self.tokenizer = init_model_and_tokenizer(
            cfg.model,
            cfg.tokenizer,
            task_tokens=[self.task_start_token],
            checkpoint_path=checkpoint_path,
        )
        self.input_info = self.model.image_encoder.input_info
        self.image_preprocess_eval = create_image_preprocessor(
            self.input_info,
            is_training=False,
            transform_type=cfg.image_transforms,
        )
        self.predictions = []
        self.references = []

    def eval_data_cfg(self):
        return chugr.DataTaskDocReadCfg(
            image_process_fn=self.image_preprocess_eval,
            text_process_fn=None,
            text_target_feat=None,
            image_input_feat=image_feature_info(self.input_info['image_fmt']),
            page_sampling=self.cfg.page_sampling,
            render_dpi=self.cfg.render_dpi,
            error_handler='log_and_continue',
        )

    def setup(self):
        self.model.to(self.device_env.device)
        self.model.eval()

    @torch.no_grad()
    def step(self, batch):
        image_input = to_device(batch['image_input'], self.device_env.device)
        prompt = self.tokenizer.convert_tokens_to_ids([self.task_start_token])
        texts = batch['text_input']
        with self.autocast():
            generated = generate_greedy(
                self.model,
                image_input,
                prompt_ids=[prompt] * len(texts),
                stop_token_ids=[self.tokenizer.eos_token_id],
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=self.cfg.max_new_tokens,
            )
        self.predictions.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        # text_input is a list (pages) of page text when not tokenized
        self.references.extend([t[0] if isinstance(t, (list, tuple)) else t for t in texts])

    def evaluate(self, loader):
        self.predictions, self.references = [], []
        for batch_idx, batch in enumerate(loader):
            self.step(batch)
        if self.device_env.world_size > 1:
            preds = self.device_env.all_gather_object(self.predictions)
            refs = self.device_env.all_gather_object(self.references)
            self.predictions = [p for r in preds for p in r]
            self.references = [p for r in refs for p in r]
        metrics = text_error_rates(self.predictions, self.references)
        metrics['num_samples'] = len(self.predictions)
        return metrics

    def save_results(self, output_dir: str, name: str = 'ocr'):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{name}_results.json'), 'w') as f:
            json.dump([{'prediction': p, 'reference': r} for p, r in zip(self.predictions, self.references)], f, indent=1)
