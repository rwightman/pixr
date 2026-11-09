import logging
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F

import chugr

from pixr.data import create_image_preprocessor
from pixr.framework import TaskTrainCfg, TaskTrain, DeviceEnv, Monitor
from pixr.generation import generate_greedy
from pixr.utils import text_error_rates
from .common import init_model_and_tokenizer, checkpoint_extra_state, image_feature_info, to_device, trim_padding

_logger = logging.getLogger(__name__)


@dataclass
class TaskCrullerPretrainCfg(TaskTrainCfg):
    text_max_length: int = 1024  # max target tokens per page
    page_sampling: str = 'random'  # page sampling for multi-page docs ('random', 'first')
    render_dpi: int = 150  # pdf render dpi (before resize to model input)
    num_metrics_samples: int = 4  # number of train samples to decode (greedy) for CER / WER train metrics
    metrics_max_new_tokens: int = 512


class TaskCrullerPretrain(TaskTrain):
    """ Cruller Pretraining Task -- read (OCR) all text on a document page, in annotation line order.

    Call sequence:
        * Task() -- task __init__() creates tokenizer, model, pre-processing
        * Initialize data-pipeline (external to Task) w/ task.train_data_cfg() to get batch / step count
        * Call setup() to pass this info back to Task and finish setting up optimizer / scheduler
        * Proceed to train by interval_start()/step() * N/interval_end(), etc
    """
    task_name = 'cruller_pretrain'
    task_start_token = '<s_pretrain>'

    def __init__(
            self,
            cfg: TaskCrullerPretrainCfg,
            device_env: DeviceEnv,
            monitor: Monitor,
            checkpoint_path: str = '',
    ):
        super().__init__(
            cfg=cfg,
            device_env=device_env,
            monitor=monitor,
        )
        self.prompt_end_token = self.task_start_token
        self.model, self.tokenizer = init_model_and_tokenizer(
            cfg.model,
            cfg.tokenizer,
            task_tokens=[self.task_start_token],
            checkpoint_path=checkpoint_path,
        )
        self.pad_token_id = self.tokenizer.pad_token_id
        self.input_info = self.model.image_encoder.input_info

        self.text_preprocess_train = chugr.create_text_preprocessor(
            self.tokenizer,
            max_length=cfg.text_max_length,
            task_start_token=self.task_start_token,
            prompt_end_token=self.prompt_end_token,
        )
        self.image_preprocess_train = create_image_preprocessor(
            self.input_info,
            is_training=True,
            transform_type=cfg.image_transforms,
        )

    def train_data_cfg(self):
        return chugr.DataTaskDocReadCfg(
            image_process_fn=self.image_preprocess_train,
            text_process_fn=self.text_preprocess_train,
            image_input_feat=image_feature_info(self.input_info['image_fmt']),
            page_sampling=self.cfg.page_sampling,
            render_dpi=self.cfg.render_dpi,
            error_handler='log_and_continue',
        )

    def extra_state(self) -> Dict[str, Any]:
        return checkpoint_extra_state(self.unwrapped_model, self.tokenizer, self.task_name)

    def _forward(self, sample: Dict[str, Any]):
        device = self.device_env.device
        image_input = to_device(sample['image_input'], device)
        text_input, text_target = trim_padding(sample['text_input'], sample['text_target'], self.pad_token_id)
        text_input = text_input[:, :-1].to(device, non_blocking=True)
        text_target = text_target[:, 1:].to(device, non_blocking=True)

        with self.autocast():
            logits = self.model(image_input, text_input)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                text_target.flatten(),
                ignore_index=-100,
                label_smoothing=self.cfg.opt.label_smoothing,
            )
        return logits, loss

    def after_step(self, sample, output, loss):
        metrics = None
        eval_data = None
        update_done = self.interval_batch_idx % self.cfg.opt.grad_accum_steps == 0
        if not update_done:
            return
        if self.metrics_frequency and self.step_idx % self.metrics_frequency == 0 and self.device_env.is_primary():
            metrics, eval_data = self.reading_metrics(sample)

        if metrics is not None or self.step_idx % self.log_frequency == 0:
            self.monitor.log_step(
                'train',
                step_idx=self.step_idx,
                step_end_idx=self.total_steps,
                interval=self.interval_idx,
                loss=loss.item(),
                lr=self.get_current_lr(),
                rate=self._get_rate(),
                metrics=metrics,
                eval_data=eval_data,
            )

    @torch.no_grad()
    def reading_metrics(self, sample):
        """ Greedy decode a few train samples and compute CER / WER vs the target text. """
        n = self.cfg.num_metrics_samples
        model = self.unwrapped_model
        was_training = model.training
        model.eval()
        image_input = sample['image_input']
        if isinstance(image_input, dict):
            image_input = {k: v[:n] for k, v in image_input.items()}
        else:
            image_input = image_input[:n]
        image_input = to_device(image_input, self.device_env.device)
        prompt_ids = self.tokenizer.convert_tokens_to_ids([self.task_start_token])
        with self.autocast():
            generated = generate_greedy(
                model,
                image_input,
                prompt_ids=[prompt_ids] * len(sample['text_input'][:n]),
                stop_token_ids=[self.tokenizer.eos_token_id],
                pad_token_id=self.pad_token_id,
                max_new_tokens=self.cfg.metrics_max_new_tokens,
            )
        predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        targets = self.tokenizer.batch_decode(sample['text_input'][:n], skip_special_tokens=True)
        # compare against the same # of target tokens that was decoded, avoid penalizing truncated decoding
        targets_trunc = []
        for t, g in zip(sample['text_input'][:n], generated):
            t = t[t != self.pad_token_id][1:]  # remove start token
            targets_trunc.append(self.tokenizer.decode(t[:max(len(g), 1)], skip_special_tokens=True))
        if was_training:
            model.train()
        rates = text_error_rates(predictions, targets_trunc)
        eval_data = {'prediction': predictions[0], 'target': targets[0]}
        _logger.info(f'Sample prediction: {predictions[0][:300]!r}')
        _logger.info(f'Sample target:     {targets[0][:300]!r}')
        return {'train_cer': rates['cer'], 'train_wer': rates['wer']}, eval_data
