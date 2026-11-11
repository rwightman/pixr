import logging
from typing import Optional, Union

import torch

from pixr.utils import load_checkpoint, clean_state_dict
from .config import ModelCfg, get_model_config
from .cruller import Cruller
from .cronut import Cronut
from .fritter import Fritter
from .text_decoder import resize_pos_embed

_logger = logging.getLogger(__name__)

_MODEL_TYPES = {
    'cruller': Cruller,
    'fritter': Fritter,
    'cronut': Cronut,
}


def create_model(
        model_name_or_cfg: Union[str, ModelCfg],
        vocab_size: int,
        pad_token_id: int = 1,
        pretrained: Optional[str] = '',
):
    """ Create a model, optionally initializing from a pixr checkpoint.

    Args:
        model_name_or_cfg: The name of the model or its configuration.
        vocab_size: Vocab size (len(tokenizer)) incl. any added task tokens.
        pad_token_id: Padding token id.
        pretrained: Path to a pixr checkpoint to initialize weights from. When the checkpoint vocab is
            smaller (tokens added for a new task), the embedding rows that exist are loaded and the rest keep
            their init. NOTE task tokens must be added to the tokenizer after the checkpoint's tokens.
    """
    if isinstance(model_name_or_cfg, str):
        model_cfg = get_model_config(model_name_or_cfg)
        assert model_cfg is not None, f'Unknown model {model_name_or_cfg}'
    else:
        assert isinstance(model_name_or_cfg, ModelCfg)
        model_cfg = model_name_or_cfg

    model = _MODEL_TYPES[model_cfg.type](model_cfg, vocab_size=vocab_size, pad_token_id=pad_token_id)

    if pretrained:
        checkpoint = load_checkpoint(pretrained)
        load_model_state(model, checkpoint.get('model', checkpoint))

    return model


@torch.no_grad()
def load_model_state(model: torch.nn.Module, state_dict, strict: bool = True):
    """ Load model weights, adapting token embedding rows if the vocab grew and learned position embeddings if the
    text length changed. """
    state_dict = dict(clean_state_dict(state_dict))
    own = model.state_dict()
    vocab_keys = getattr(model, 'vocab_param_names', ('text_decoder.embed_tokens.weight', 'text_decoder.output_bias'))
    for key in vocab_keys:
        if key in state_dict and key in own and state_dict[key].shape != own[key].shape:
            src = state_dict[key]
            dst = own[key].clone()
            assert src.shape[0] <= dst.shape[0], \
                f'Checkpoint vocab ({src.shape[0]}) larger than model vocab ({dst.shape[0]}).'
            dst[:src.shape[0]] = src
            dst[src.shape[0]:] = src.mean(0, keepdim=True)
            state_dict[key] = dst
            _logger.info(f'Expanded {key} {src.shape[0]} -> {dst.shape[0]}.')
    for key, src in list(state_dict.items()):
        # learned (BART) position embeddings, the text length was changed from the checkpoint's
        if key.endswith('embed_positions.weight') and key in own and src.shape != own[key].shape:
            state_dict[key] = resize_pos_embed(src, own[key].shape[0])
            _logger.info(f'Resized {key} {src.shape[0]} -> {own[key].shape[0]}.')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f'Error loading model state. Missing: {missing}, unexpected: {unexpected}')
    return missing, unexpected
