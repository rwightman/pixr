import logging
from collections import OrderedDict
from dataclasses import replace
from typing import Dict, Optional, Tuple, Union

import timm
import torch
from torch import nn as nn
from timm.models import adapt_input_conv

from chugr.common import image_mode_to_chs

from .config import ImageEncoderCfg

_logger = logging.getLogger(__name__)


def _load_pretrained_from(model: nn.Module, source_name: str, in_chans: int):
    """ Initialize `model` from the pretrained weights of a different timm model.

    Used to init a NaFlexVit from a fixed resolution ViT (e.g. CLIP) checkpoint.
    """
    source = timm.create_model(source_name, pretrained=True, num_classes=0)
    state_dict = source.state_dict()
    # adapt patch embed input channels (e.g. RGB -> grayscale) before any conv -> linear conversion
    for k in ('patch_embed.proj.weight', 'embeds.proj.weight'):
        if k in state_dict and state_dict[k].ndim == 4 and state_dict[k].shape[1] != in_chans:
            state_dict[k] = adapt_input_conv(in_chans, state_dict[k])
    if isinstance(model, timm.models.NaFlexVit):
        from timm.models.naflexvit import checkpoint_filter_fn
        state_dict = checkpoint_filter_fn(state_dict, model)
    # drop head / pooling weights, not used here
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith(('head.', 'attn_pool.', 'fc_norm.'))}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        _logger.warning(f'Missing keys when initializing from {source_name}: {missing}')
    if unexpected:
        _logger.warning(f'Unexpected keys when initializing from {source_name}: {unexpected}')
    return source.pretrained_cfg


class ImageEncoderTimm(nn.Module):
    """ timm image encoder producing a token sequence (+ optional validity mask) for the text decoder. """

    def __init__(self, cfg: ImageEncoderCfg, out_dim: Optional[int] = None):
        super().__init__()
        assert cfg.name
        self.is_naflex = cfg.is_naflex
        in_chans = image_mode_to_chs(cfg.image_fmt)

        timm_kwargs = dict(cfg.model_kwargs)
        if cfg.image_size is not None and not self.is_naflex:
            timm_kwargs.setdefault('img_size', tuple(cfg.image_size))
        if cfg.drop_path_rate is not None:
            timm_kwargs['drop_path_rate'] = cfg.drop_path_rate
        if cfg.patch_drop_rate is not None:
            timm_kwargs['patch_drop_rate'] = cfg.patch_drop_rate
        if cfg.attn_backend is not None:
            assert self.is_naflex, 'attn_backend is only supported for NaFlexVit encoders.'
            timm_kwargs['attn_backend'] = cfg.attn_backend

        load_from_hub = cfg.pretrained and not cfg.pretrained_from
        if cfg.name == 'naflexvit':
            # generic NaFlexVit, architecture fully specified by model_kwargs (NaFlexVitCfg fields)
            from timm.models.naflexvit import NaFlexVit, NaFlexVitCfg
            assert not load_from_hub, "Generic 'naflexvit' has no pretrained weights, use pretrained_from."
            naflex_cfg = NaFlexVitCfg(**{**timm_kwargs, 'global_pool': ''})
            self.trunk = NaFlexVit(cfg=naflex_cfg, in_chans=in_chans, num_classes=0)
        else:
            # create w/ default pooling so pretrained weights load strictly, then remove pooling & classifier
            # NOTE timm adapts pretrained (conv or NaFlex linear) patch embed weights to in_chans on load
            if self.is_naflex and load_from_hub and in_chans != 3:
                import timm.models._manipulate as timm_manipulate
                if not hasattr(timm_manipulate, 'adapt_input_linear_patch'):
                    raise RuntimeError(
                        f'timm {timm.__version__} cannot adapt pretrained NaFlexVit linear patch embed weights to '
                        f'in_chans={in_chans}, a timm w/ adapt_input_linear_patch (naflex-attn-backends) is required.')
            self.trunk = timm.create_model(
                cfg.name,
                pretrained=load_from_hub,
                in_chans=in_chans,
                num_classes=0,
                **timm_kwargs
            )
            self.trunk.reset_classifier(0, global_pool='')
        pretrained_cfg = getattr(self.trunk, 'pretrained_cfg', {})
        if cfg.pretrained and cfg.pretrained_from:
            pretrained_cfg = _load_pretrained_from(self.trunk, cfg.pretrained_from, in_chans)

        self.num_prefix_tokens = getattr(self.trunk, 'num_prefix_tokens', 0)
        prev_dim = self.trunk.num_features
        out_dim = cfg.out_dim or out_dim or prev_dim
        head_type = cfg.head_type
        if head_type is None and out_dim != prev_dim:
            head_type = 'linear'
        if head_type:
            assert head_type in ('linear', 'mlp')
            if head_type == 'mlp':
                self.head = timm.layers.Mlp(
                    in_features=prev_dim,
                    hidden_features=int(prev_dim * 2),
                    drop=(cfg.drop_rate or 0., 0.),
                    out_features=out_dim,
                )
            else:
                self.head = nn.Sequential(OrderedDict([
                    ('drop', nn.Dropout(cfg.drop_rate) if cfg.drop_rate else nn.Identity()),
                    ('proj', nn.Linear(prev_dim, out_dim)),
                ]))
        else:
            self.head = nn.Identity()
        self.out_dim = out_dim

        mean = cfg.image_mean or pretrained_cfg.get('mean', (0.5,) * 3)
        std = cfg.image_std or pretrained_cfg.get('std', (0.5,) * 3)
        if in_chans == 1 and len(mean) > 1:
            mean = (sum(mean) / len(mean),)
            std = (sum(std) / len(std),)
        patch_size = None
        if self.is_naflex:
            patch_size = self.trunk.embeds.patch_size
            patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size
        # resolved config (w/ normalization constants) so the model can be rebuilt w/o pretrained weights
        self.cfg = replace(cfg, image_mean=tuple(mean), image_std=tuple(std))
        self.input_info = dict(
            image_fmt=cfg.image_fmt,
            image_size=tuple(cfg.image_size) if cfg.image_size is not None else None,
            max_seq_len=cfg.max_seq_len,
            patch_size=patch_size,
            mean=tuple(mean),
            std=tuple(std),
            naflex=self.is_naflex,
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.trunk.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = self.trunk.no_weight_decay() if hasattr(self.trunk, 'no_weight_decay') else set()
        return {'trunk.' + n for n in nwd}

    def forward(
            self,
            x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            tokens: [B, N, C] encoder output tokens
            mask: [B, N] bool validity mask (True == valid), None if all tokens are valid
        """
        mask = None
        if isinstance(x, dict):
            assert self.is_naflex, 'Dict (patch sequence) input requires a NaFlex model.'
            # NOTE patch_valid may be None (or absent) for a batch w/o padding, no masks are built then
            patch_valid = x.get('patch_valid', None)
            out = self.trunk.forward_features(
                patches=x['patches'],
                patch_coord=x['patch_coord'],
                patch_valid=patch_valid,
            )
            tokens = out['patches']
            valid = out.get('patch_valid', None)
            if valid is None:
                valid = patch_valid
            if valid is not None:
                if self.num_prefix_tokens:
                    prefix_valid = valid.new_ones(valid.shape[0], self.num_prefix_tokens)
                    valid = torch.cat([prefix_valid, valid], dim=1)
                mask = valid.bool()
        else:
            tokens = self.trunk.forward_features(x)
            if tokens.ndim == 4:
                # flatten spatial feature maps (convnets, hybrids w/o tokens), NCHW or NHWC
                if getattr(self.trunk, 'output_fmt', 'NCHW') == 'NHWC':
                    tokens = tokens.flatten(1, 2)
                else:
                    tokens = tokens.flatten(2).transpose(1, 2)
        tokens = self.head(tokens)
        return tokens, mask
