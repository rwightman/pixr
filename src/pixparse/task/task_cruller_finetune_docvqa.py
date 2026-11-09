import logging
from dataclasses import dataclass
from typing import Any, Dict

import torch.nn.functional as F

import chug

from pixparse.data import create_image_preprocessor
from pixparse.framework import DeviceEnv, Monitor, TaskTrain, TaskTrainCfg
from .common import init_model_and_tokenizer, checkpoint_extra_state, to_device, trim_padding
from .task_cruller_eval_docvqa import DOCVQA_START_TOKEN, DOCVQA_TOKENS, DocVqaEvaluator, docvqa_task_cfg, \
    evaluate_docvqa

_logger = logging.getLogger(__name__)


@dataclass
class TaskCrullerFinetuneDocVqaCfg(TaskTrainCfg):
    text_max_length: int = 128
    eval_max_new_tokens: int = 64


class TaskCrullerFinetuneDocVqa(TaskTrain):
    """ DocVQA fine-tuning, Donut style prompt / answer format.

    '<s_docvqa><s_question>{question}</s_question><s_answer>{answer}</s_answer></s>'
    Loss is only applied to the answer tokens (after '<s_answer>').
    """
    task_name = 'cruller_finetune_docvqa'
    task_start_token = DOCVQA_START_TOKEN
    prompt_end_token = '<s_answer>'

    def __init__(
            self,
            cfg: TaskCrullerFinetuneDocVqaCfg,
            device_env: DeviceEnv,
            monitor: Monitor = None,
            checkpoint_path: str = '',
    ):
        super().__init__(
            cfg=cfg,
            device_env=device_env,
            monitor=monitor,
        )
        self.model, self.tokenizer = init_model_and_tokenizer(
            cfg.model,
            cfg.tokenizer,
            task_tokens=DOCVQA_TOKENS,
            checkpoint_path=checkpoint_path,
        )
        self.pad_token_id = self.tokenizer.pad_token_id
        self.input_info = self.model.image_encoder.input_info

        self.text_preprocess_train = chug.create_text_preprocessor(
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
        self.image_preprocess_eval = create_image_preprocessor(
            self.input_info,
            is_training=False,
            transform_type=cfg.image_transforms,
        )
        self.evaluator = DocVqaEvaluator(
            self.tokenizer,
            device=device_env.device,
            autocast=self.autocast,
            max_new_tokens=cfg.eval_max_new_tokens,
        )

    def train_data_cfg(self):
        return docvqa_task_cfg(
            self.image_preprocess_train,
            self.input_info['image_fmt'],
            text_process_fn=self.text_preprocess_train,
            is_training=True,
        )

    def eval_data_cfg(self):
        return docvqa_task_cfg(self.image_preprocess_eval, self.input_info['image_fmt'], is_training=False)

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

    def evaluate(self, loader):
        return evaluate_docvqa(self.unwrapped_model, loader, self.evaluator, self.device_env)
