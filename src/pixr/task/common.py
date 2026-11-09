import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from chugr.common import ImageFeatureInfo
from chugr.doc import DEFAULT_DOC_KEY

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
