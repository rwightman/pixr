""" Cross-attention backends for padded (variable length) encoder sequences.

* 'sdpa': F.scaled_dot_product_attention w/ a [B, 1, 1, S] bool key mask
* 'varlen': valid encoder tokens are packed once (before the K / V projections) and attention runs w/
  torch varlen (Flash / cuDNN) kernels, no compute is spent on padding
* 'flex': FlexAttention w/ a padding block mask (compiled), single token decode steps use 'sdpa'

`prepare_cross_attn` returns the encoder states to project K / V from (packed for varlen) and a CrossAttnMask
that is passed to every cross-attention layer.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from timm.layers.attention_backends import (
    VarlenMeta, build_varlen_meta, pack_tokens, has_varlen, has_flex, _get_compiled_flex
)

if has_varlen:
    from torch.nn.attention.varlen import varlen_attn
if has_flex:
    from torch.nn.attention.flex_attention import create_block_mask

CROSS_ATTN_BACKENDS = ('sdpa', 'varlen', 'flex')


class CrossAttnMask:
    """ Cross-attention key mask of a batch of (padded) encoder sequences.

    A plain class (not a dataclass), FSDP2 input casting rebuilds dataclasses passed into a wrapped layer, the
    lazily built flex block masks cached here would be built on a copy and lost (rebuilt every layer / step).
    """
    __slots__ = ('backend', 'key_valid', 'varlen', '_flex_masks')

    def __init__(self, backend: str, key_valid: torch.Tensor, varlen: Optional[VarlenMeta] = None):
        self.backend = backend
        self.key_valid = key_valid  # [B, S] bool validity of encoder tokens (padded layout)
        self.varlen = varlen  # packing metadata when K / V are packed
        self._flex_masks: Dict[int, object] = {}

    @property
    def batch_size(self) -> int:
        return self.key_valid.shape[0]

    def sdpa_mask(self) -> torch.Tensor:
        return self.key_valid[:, None, None, :]

    def flex_mask(self, q_len: int):
        if q_len not in self._flex_masks:
            valid = self.key_valid
            B, S = valid.shape
            self._flex_masks[q_len] = create_block_mask(
                lambda b, h, q_idx, kv_idx: valid[b, kv_idx], B, None, q_len, S, device=valid.device)
        return self._flex_masks[q_len]


def _varlen_supported(x: torch.Tensor) -> bool:
    if not has_varlen or x.device.type != 'cuda':
        return False
    return x.dtype in (torch.float16, torch.bfloat16) or torch.is_autocast_enabled('cuda')


def prepare_cross_attn(
        encoder_hidden_states: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        backend: str = 'sdpa',
) -> Tuple[torch.Tensor, Optional[CrossAttnMask]]:
    """ Returns (states to project cross-attn K / V from, cross-attn mask object or None). """
    assert backend in CROSS_ATTN_BACKENDS, f'Unknown cross-attention backend {backend}'
    if encoder_mask is None:
        return encoder_hidden_states, None  # no padding, plain sdpa
    key_valid = encoder_mask.bool()
    if backend == 'varlen' and _varlen_supported(encoder_hidden_states):
        meta = build_varlen_meta(key_valid)
        return pack_tokens(encoder_hidden_states, meta), CrossAttnMask('varlen', key_valid, varlen=meta)
    if backend == 'flex' and has_flex:
        return encoder_hidden_states, CrossAttnMask('flex', key_valid)
    return encoder_hidden_states, CrossAttnMask('sdpa', key_valid)


def cross_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[CrossAttnMask],
        dropout_p: float = 0.,
) -> torch.Tensor:
    """ q [B, H, T, D], k / v [B, H, S, D] (or [1, H, S_packed, D] for varlen) -> [B, H, T, D] """
    if mask is None:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    if mask.backend == 'varlen':
        assert dropout_p == 0., 'attention dropout not supported w/ varlen cross-attention'
        B, H, T, D = q.shape
        meta = mask.varlen
        cu_q = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=q.device)
        out = varlen_attn(
            q.transpose(1, 2).reshape(B * T, H, D),
            k[0].transpose(0, 1),
            v[0].transpose(0, 1),
            cu_q,
            meta.cu_seqlens,
            T,
            meta.max_seqlen,
        )
        return out.reshape(B, T, H, D).transpose(1, 2)
    # NOTE single token decode steps take sdpa, but w/ several hypotheses per image (beams) folded into the query
    # length a step has q_len == group size and takes flex (a block mask per q_len), sdpa may be faster for so few
    # queries, not benchmarked
    if mask.backend == 'flex' and q.shape[2] > 1 and dropout_p == 0.:
        return _get_compiled_flex()(q, k, v, block_mask=mask.flex_mask(q.shape[2]))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask.sdpa_mask(), dropout_p=dropout_p)


SELF_ATTN_BACKENDS = ('sdpa', 'flex')


def build_prefix_lm_block_mask(
        key_valid: torch.Tensor,
        prefix_len: int,
        window: Optional[int] = None,
        text_sees_full_prefix: bool = True,
):
    """ FlexAttention block mask for prefix-LM self-attention over [prefix (P) ; text] sequences.

    Allowed pairs: prefix <-> prefix (bidirectional), text -> text (causal), text -> prefix. Padding keys are masked
    (key_valid [B, S]), every query attends to itself so no row is fully masked.

    Args:
        key_valid: [B, S] bool key validity
        prefix_len: length of the (bidirectional) prefix block, 0 for plain causal
        window: optional sliding window (local attention), symmetric within the prefix, causal for text
        text_sees_full_prefix: w/ a window, text queries still attend to the whole prefix
    """
    assert has_flex
    B, S = key_valid.shape
    valid = key_valid.contiguous()
    P = int(prefix_len)

    def mask_mod(b, h, q_idx, kv_idx):
        allowed = (kv_idx <= q_idx) | ((q_idx < P) & (kv_idx < P))
        if window is not None:
            local = (kv_idx > q_idx - window) & (kv_idx < q_idx + window)
            if text_sees_full_prefix and P > 0:
                local = local | ((kv_idx < P) & (q_idx >= P))
            allowed = allowed & local
        return (allowed & valid[b, kv_idx]) | (q_idx == kv_idx)

    return create_block_mask(mask_mod, B, None, S, S, device=valid.device)


def self_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask=None,
        is_causal: bool = False,
        dropout_p: float = 0.,
        enable_gqa: bool = False,
) -> torch.Tensor:
    """ Self-attention dispatch, FlexAttention when attn_mask is a BlockMask, else sdpa. """
    if has_flex and _is_block_mask(attn_mask):
        assert dropout_p == 0., 'attention dropout not supported w/ flex attention'
        return _get_compiled_flex()(q, k, v, block_mask=attn_mask, enable_gqa=enable_gqa)
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal and attn_mask is None,
        enable_gqa=enable_gqa,
    )


def _is_block_mask(x) -> bool:
    if not has_flex:
        return False
    from torch.nn.attention.flex_attention import BlockMask
    return isinstance(x, BlockMask)
