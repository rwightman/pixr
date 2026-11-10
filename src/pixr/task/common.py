import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from chugr.common import ImageFeatureInfo
from chugr.doc import DEFAULT_DOC_KEY

from pixr.data import create_image_preprocessor, ImagePreprocessorFactory
from pixr.framework import DeviceEnv, TaskCfg, TrainingTask
from pixr.framework.task import move_to_device
from pixr.generation import generate_greedy
from pixr.loss import fused_linear_cross_entropy, lm_cross_entropy
from pixr.models import ModelArgs, ModelCfg, Cruller, create_model, load_model_state
from pixr.tokenizers import TokenizerCfg, create_tokenizer, get_added_tokens
from pixr.utils import load_checkpoint

_logger = logging.getLogger(__name__)


def init_model_and_tokenizer(
        model_args: ModelArgs,
        tokenizer_cfg: Optional[TokenizerCfg],
        task_tokens: Sequence[str],
        checkpoint_path: str = '',
) -> Tuple[Cruller, Any]:
    """ Create tokenizer (w/ task tokens) and model, optionally initialized from a pixr checkpoint.

    When a checkpoint is given, its model config (architecture) and added tokens are used. The checkpoint's
    tokens are added before the task tokens so existing token embedding rows line up, new task tokens are
    initialized from the mean embedding. `model_args.name` is ignored, other overrides (e.g. max_seq_len)
    are applied.
    """
    checkpoint = None
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint is not None and 'model_cfg' in checkpoint:
        model_cfg = ModelCfg.from_dict(checkpoint['model_cfg'])
        model_cfg = model_args.apply_overrides(model_cfg, from_checkpoint=True)
        _logger.info(f'Using model config from checkpoint {checkpoint_path}.')
    else:
        model_cfg = model_args.resolve()

    if tokenizer_cfg is None:
        name = checkpoint.get('tokenizer') if checkpoint is not None else None
        tokenizer_cfg = TokenizerCfg(name=name or model_cfg.text_decoder.name)
    checkpoint_tokens = list(checkpoint.get('tokens', [])) if checkpoint is not None else []
    tokenizer = create_tokenizer(tokenizer_cfg, added_tokens=checkpoint_tokens + list(task_tokens))

    model = create_model(model_cfg, vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id)
    if checkpoint is not None:
        load_model_state(model, checkpoint.get('model', checkpoint))
    return model, tokenizer


def checkpoint_extra_state(model: Cruller, tokenizer, task_name: str) -> Dict[str, Any]:
    return {
        'model_cfg': model.cfg.to_dict(),
        'tokenizer': tokenizer.name_or_path,
        'tokens': get_added_tokens(tokenizer),
        'task': task_name,
    }


def image_feature_info(image_fmt: str) -> ImageFeatureInfo:
    return ImageFeatureInfo('image_input', input_key=DEFAULT_DOC_KEY, image_mode=image_fmt)


def to_device(x: Union[torch.Tensor, Dict[str, torch.Tensor]], device, non_blocking=True):
    if isinstance(x, dict):
        return {k: v.to(device, non_blocking=non_blocking) for k, v in x.items()}
    return x.to(device, non_blocking=non_blocking)


def trim_padding(text_input: torch.Tensor, text_target: torch.Tensor, pad_token_id: int):
    """ Trim batch sequence length to the longest non-pad sequence. """
    max_len = int((text_input != pad_token_id).sum(1).max().item())
    return text_input[:, :max_len], text_target[:, :max_len]


class ImageTextTask(TrainingTask):
    """ Base for image -> text tasks, agnostic to the model family (Cruller, Fritter, Cronut).

    Models implement: forward(image_input, text_input) -> text logits, prefill() / step() for generation,
    image_encoder.input_info (image preprocessing requirements), cfg, no_weight_decay(), set_grad_checkpointing().
    """
    task_tokens: Sequence[str] = ()

    def __init__(self, cfg: TaskCfg, device_env: DeviceEnv, checkpoint_path: str = ''):
        model, tokenizer = init_model_and_tokenizer(
            cfg.model,
            cfg.tokenizer,
            task_tokens=self.task_tokens,
            checkpoint_path=checkpoint_path,
        )
        super().__init__(cfg, model, device_env)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.input_info = model.image_encoder.input_info
        self.image_preprocess_train = create_image_preprocessor(
            self.input_info, is_training=True, transform_type=cfg.image_transforms)
        self.image_preprocess_eval = create_image_preprocessor(
            self.input_info, is_training=False, transform_type=cfg.image_transforms)
        self.image_preprocess_factory = ImagePreprocessorFactory(self.input_info, True, cfg.image_transforms)

    def extra_state(self) -> Dict[str, Any]:
        return checkpoint_extra_state(self.trainable_module, self.tokenizer, self.task_name)

    def lm_loss(self, batch: Dict[str, Any]):
        """ Teacher forced next token cross-entropy on text targets (masked w/ -100), optional z-loss.

        Returns (outputs, losses), w/ the fused (chunked) LM head + CE the full logits are never materialized.
        """
        text_input, text_target = trim_padding(batch['text_input'], batch['text_target'], self.pad_token_id)
        model = self.trainable_module
        target = text_target[:, 1:]
        z_loss = self.cfg.z_loss_weight > 0
        outputs = {}
        if self.cfg.loss_chunk_size > 0:
            hidden = model(batch['image_input'], text_input[:, :-1], return_hidden=True)
            weight, bias = model.lm_head_params()
            ce, z = fused_linear_cross_entropy(
                hidden.flatten(0, 1),
                weight,
                target.flatten(),
                bias=bias,
                chunk_size=self.cfg.loss_chunk_size,
                z_loss=z_loss,
                label_smoothing=self.cfg.label_smoothing,
            )
        else:
            outputs['logits'] = model(batch['image_input'], text_input[:, :-1])
            ce, z = lm_cross_entropy(
                outputs['logits'], target, z_loss=z_loss, label_smoothing=self.cfg.label_smoothing)
        losses = {'loss': ce}
        if z is not None:
            losses['loss'] = ce + self.cfg.z_loss_weight * z
            losses['ce_loss'] = ce
            losses['z_loss'] = z
        return outputs, losses

    def training_forward(self, batch: Dict[str, Any]):
        return self.lm_loss(batch)

    @torch.no_grad()
    def generate(self, image_input, prompts: Sequence[Sequence[int]], stop_token_ids, max_new_tokens: int,
                 use_ema: bool = False):
        model = self.get_trainable_module(use_ema)
        was_training = model.training
        model.eval()
        with self.autocast():
            out = generate_greedy(
                model,
                move_to_device(image_input, self.device),
                prompt_ids=prompts,
                stop_token_ids=stop_token_ids,
                pad_token_id=self.pad_token_id,
                max_new_tokens=max_new_tokens,
            )
        if was_training:
            model.train()
        return out
