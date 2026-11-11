import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from chugr.common import ImageFeatureInfo
from chugr.doc import DEFAULT_DOC_KEY

from pixr.data import create_image_preprocessor, ImagePreprocessorFactory
from pixr.framework import DeviceEnv, TaskCfg, TrainingTask
from pixr.framework.task import move_to_device
from pixr.generation import generate_beam, generate_greedy
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


def drop_full_patch_valid(image_input):
    """ NaFlex (patch dict) input, drop the patch validity mask of a batch w/o padding (None == all valid).

    Models then skip padding masks entirely (unmasked attention can use the fastest kernels, e.g. fixed resolution
    inference). Only checked while the mask is on the CPU (before the device transfer), so there is no GPU sync
    or data dependent branch in the model.
    """
    if isinstance(image_input, dict):
        valid = image_input.get('patch_valid')
        if isinstance(valid, torch.Tensor) and valid.device.type == 'cpu' and bool(valid.all()):
            image_input = {**image_input, 'patch_valid': None}
    return image_input


def trim_padding(text_input: torch.Tensor, text_target: torch.Tensor, pad_token_id: int, multiple: int = 1):
    """ Trim batch sequence length to the longest non-pad sequence (right padded), rounded up to `multiple`.

    NOTE reads the length (a sync for device tensors), call on CPU batches before the device transfer.
    """
    max_len = int((text_input != pad_token_id).sum(1).max().item())
    if multiple > 1:
        max_len = -(-max_len // multiple) * multiple
    max_len = min(max(max_len, 1), text_input.shape[1])
    return text_input[:, :max_len], text_target[:, :max_len]


class ImageTextTask(TrainingTask):
    """ Base for image -> text tasks, agnostic to the model family (Cruller, Fritter, Cronut).

    Models implement: forward(image_input, text_input) -> text logits, the generation interface (encode_image() /
    prefill() / step() w/ a reorderable DecodeState and max_text_length, see pixr.generation), image_encoder.input_info
    (image preprocessing requirements), cfg, no_weight_decay(), set_grad_checkpointing().
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
        return checkpoint_extra_state(self.get_trainable_module(), self.tokenizer, self.task_name)

    def lm_loss(self, batch: Dict[str, Any]):
        """ Teacher forced next token cross-entropy on text targets (masked w/ -100), optional z-loss.

        Returns (outputs, losses), w/ the fused (chunked) LM head + CE the full logits are never materialized.
        """
        # NOTE padding is trimmed in prepare_batch (on the CPU), no length sync / data dependent shape logic here
        text_input, text_target = batch['text_input'], batch['text_target']
        model = self.trainable_module  # possibly DDP / compile wrapped, for the forward
        target = text_target[:, 1:]
        z_loss = self.cfg.z_loss_weight > 0
        outputs = {}
        if self.cfg.loss_chunk_size > 0:
            hidden = model(batch['image_input'], text_input[:, :-1], return_hidden=True)
            # outside the model forward, w/ FSDP2 the root's params (incl. the LM head / tied embedding) stay
            # unsharded after forward until backward
            weight, bias = self.get_trainable_module().lm_head_params()
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

    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # CPU side batch prep before the device transfer, keeps length / validity checks (syncs) out of the step
        batch = dict(batch)
        if 'image_input' in batch:
            batch['image_input'] = drop_full_patch_valid(batch['image_input'])
        text_input, text_target = batch.get('text_input'), batch.get('text_target')
        if isinstance(text_input, torch.Tensor) and isinstance(text_target, torch.Tensor):
            # text is padded to max_length per sample by the data pipeline, trim to the batch
            batch['text_input'], batch['text_target'] = trim_padding(
                text_input, text_target, self.pad_token_id, multiple=self.cfg.text_pad_multiple)
        return super().prepare_batch(batch)

    @torch.no_grad()
    def generate(self, image_input, prompts: Sequence[Sequence[int]], stop_token_ids, max_new_tokens: int,
                 use_ema: bool = False, num_beams: int = 1, length_penalty: float = 1.0):
        """ Greedy (num_beams == 1) or beam search decode. Under FSDP call within `full_params(use_ema)`. """
        model = self.get_trainable_module(use_ema)
        if not self.has_full_params(model):
            # W/ sharded params every prefill / step would all-gather, ranks decode different numbers of steps
            # (per rank stopping, uneven eval batches) and would hang waiting on each other.
            raise RuntimeError(
                f'Generation under FSDP requires `with task.full_params(use_ema={use_ema}):` around it '
                f'(for the {"EMA" if use_ema else "model"} weights being generated from).')
        was_training = model.training
        model.eval()
        kwargs = dict(
            prompt_ids=prompts,
            stop_token_ids=stop_token_ids,
            pad_token_id=self.pad_token_id,
            max_new_tokens=max_new_tokens,
            # ids the output head covers but the tokenizer has no token for (tiktoken), never generated
            bad_token_ids=getattr(self.tokenizer, 'unused_token_ids', None) or None,
        )
        image_input = move_to_device(drop_full_patch_valid(image_input), self.device)
        with self.autocast():
            if num_beams > 1:
                out = generate_beam(model, image_input, num_beams=num_beams, length_penalty=length_penalty, **kwargs)
            else:
                out = generate_greedy(model, image_input, **kwargs)
        if was_training:
            model.train()
        return out
