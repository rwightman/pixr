""" Fritter: image encoder + projector -> image token prefix for a decoder-only (causal) LM w/ prefix-LM attention.

Sequence: [image tokens (bidirectional)] [text: task start / prompt ... answer (causal)]

The text tokens are the same as for Cruller tasks (task start token, prompt, prompt end token, answer, eos), so
tasks, losses, and evaluation are shared. Forward returns logits for the text positions only.
"""
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelCfg
from .cruller import create_text_decoder
from .image_encoder_timm import ImageEncoderTimm


def merge_patch_tokens(
        tokens: torch.Tensor,
        valid: torch.Tensor,
        patch_coord: torch.Tensor,
        merge: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Merge (pixel-unshuffle) m x m neighbouring patch tokens of variable size (NaFlex) grids.

    Args:
        tokens: [B, N, C] patch tokens (no prefix tokens)
        valid: [B, N] bool validity
        patch_coord: [B, N, 2] (y, x) patch coordinates
        merge: merge factor m

    Returns:
        merged tokens [B, M, m * m * C] in row-major order of the merged grid, validity [B, M]
    """
    B, N, C = tokens.shape
    y = patch_coord[..., 0].long()
    x = patch_coord[..., 1].long()
    my, mx = y // merge, x // merge
    w_merged = int(mx.max().item()) + 1
    h_merged = int(my.max().item()) + 1
    num_slots = h_merged * w_merged
    slot = my * w_merged + mx  # merged token index
    sub = (y % merge) * merge + (x % merge)  # position within merged token
    flat = slot * (merge * merge) + sub
    flat = torch.where(valid, flat, torch.full_like(flat, num_slots * merge * merge))  # dump invalid in extra slot
    out = tokens.new_zeros(B, num_slots * merge * merge + 1, C)
    out.scatter_(1, flat[..., None].expand(-1, -1, C), tokens)
    out = out[:, :-1].reshape(B, num_slots, merge * merge * C)
    out_valid = torch.zeros(B, num_slots * merge * merge + 1, dtype=torch.bool, device=tokens.device)
    out_valid.scatter_(1, flat, valid)
    out_valid = out_valid[:, :-1].reshape(B, num_slots, merge * merge).any(-1)
    # compact valid merged tokens to the front (stable, keeps row-major order) and trim
    order = torch.argsort((~out_valid).to(torch.int8), dim=1, stable=True)
    out = torch.gather(out, 1, order[..., None].expand(-1, -1, out.shape[-1]))
    out_valid = torch.gather(out_valid, 1, order)
    max_valid = int(out_valid.sum(1).max().item())
    return out[:, :max_valid], out_valid[:, :max_valid]


class Projector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, proj_type: str = 'mlp'):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        if proj_type == 'mlp':
            self.fc1 = nn.Linear(in_dim, out_dim)
            self.fc2 = nn.Linear(out_dim, out_dim)
        else:
            assert proj_type == 'linear'
            self.fc1 = nn.Linear(in_dim, out_dim)
            self.fc2 = None

    def forward(self, x):
        x = self.fc1(self.norm(x))
        if self.fc2 is not None:
            x = self.fc2(F.gelu(x))
        return x


class Fritter(nn.Module):
    def __init__(self, cfg: ModelCfg, vocab_size: int, pad_token_id: int = 0):
        super().__init__()
        assert cfg.text_decoder.type == 'ettin', 'Fritter currently requires an Ettin (decoder-only) text model.'
        dec_cfg = replace(cfg.text_decoder, cross_attn_every=0)
        self.text_decoder = create_text_decoder(dec_cfg, vocab_size=vocab_size, pad_token_id=pad_token_id)
        self.image_encoder = ImageEncoderTimm(replace(cfg.image_encoder, head_type=None, out_dim=None))
        self.token_merge = cfg.prefix_token_merge
        merge_dim = self.image_encoder.out_dim * self.token_merge ** 2
        self.projector = Projector(merge_dim, self.text_decoder.dim, cfg.projector_type)
        self.cfg = replace(cfg, image_encoder=self.image_encoder.cfg, text_decoder=self.text_decoder.cfg)

    @property
    def max_text_length(self) -> int:
        return self.text_decoder.max_length

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.image_encoder.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd = set()
        no_wd |= {'image_encoder.' + n for n in self.image_encoder.no_weight_decay()}
        no_wd |= {'text_decoder.' + n for n in self.text_decoder.no_weight_decay()}
        return no_wd

    def encode(
            self,
            image_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Image -> prefix embeddings [B, P, C_text] + validity mask [B, P]. """
        tokens, mask = self.image_encoder(image_input)
        if mask is None:
            mask = tokens.new_ones(tokens.shape[:2], dtype=torch.bool)
        if self.token_merge > 1:
            assert isinstance(image_input, dict), 'Token merging currently requires NaFlex (patch dict) input.'
            num_prefix = self.image_encoder.num_prefix_tokens
            tokens, mask = merge_patch_tokens(
                tokens[:, num_prefix:],
                mask[:, num_prefix:],
                image_input['patch_coord'],
                self.token_merge,
            )
        return self.projector(tokens), mask

    def forward(
            self,
            image_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
            text_input: torch.Tensor,
    ) -> torch.Tensor:
        """ Teacher forced forward, returns text logits [B, T, V]. """
        prefix, prefix_mask = self.encode(image_input)
        return self.text_decoder(text_input, prefix_embeds=prefix, prefix_mask=prefix_mask)

    def prefill(self, image_input, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor):
        prefix, prefix_mask = self.encode(image_input)
        cache = [{} for _ in self.text_decoder.layers]
        attn_mask = torch.cat([prefix_mask, prompt_mask], dim=1)
        position_ids = (attn_mask.long().cumsum(1) - 1).clamp(min=0)
        logits = self.text_decoder(
            prompt_ids,
            prefix_embeds=prefix,
            position_ids=position_ids,
            attention_mask=attn_mask,
            cache=cache,
        )
        state = dict(cache=cache, attn_mask=attn_mask, pos=position_ids[:, -1])
        return logits[:, -1], state

    def step(self, token_ids: torch.Tensor, state: Dict[str, Any]):
        state['pos'] = state['pos'] + 1
        state['attn_mask'] = torch.cat([state['attn_mask'], state['attn_mask'].new_ones(token_ids.shape)], dim=1)
        logits = self.text_decoder(
            token_ids,
            position_ids=state['pos'][:, None],
            attention_mask=state['attn_mask'],
            cache=state['cache'],
        )
        return logits[:, -1], state
