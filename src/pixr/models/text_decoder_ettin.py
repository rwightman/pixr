""" ModernBERT style (Ettin) decoder w/ added cross-attention

Pre-norm transformer w/ RoPE, GeGLU MLP, bias-free LayerNorms, alternating global / sliding window self-attention,
weights loaded from Hugging Face Ettin decoder checkpoints (e.g. jhu-clsp/ettin-decoder-150m) by name.

Cross-attention does not exist in the pretrained decoder, a pre-norm cross-attention sub-block (zero init output
projection) is inserted after self-attention in every `cross_attn_every` layer so the decoder starts as the
pretrained LM.
"""
import json
import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import CrossAttnMask, prepare_cross_attn, build_prefix_lm_block_mask, self_attention, has_flex
from .config import TextDecoderCfg
from .text_decoder import Attention

_logger = logging.getLogger(__name__)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D], cos / sin: [B, 1, T, D]
    dtype = x.dtype
    x = x.float()
    return (x * cos + rotate_half(x) * sin).to(dtype)


class RopeSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, rope, attn_mask=None, is_causal=False, cache=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = rope
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if cache is not None:
            if 'k' in cache:
                k = torch.cat([cache['k'], k], dim=2)
                v = torch.cat([cache['v'], v], dim=2)
            cache['k'], cache['v'] = k, v
        x = self_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.attn_drop if self.training else 0.,
        )
        return self.Wo(x.transpose(1, 2).reshape(B, T, C))


class GeGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.):
        super().__init__()
        self.Wi = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.Wo = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = drop

    def forward(self, x):
        x, gate = self.Wi(x).chunk(2, dim=-1)
        x = F.gelu(x) * gate
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.Wo(x)


class EttinDecoderLayer(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            ffn_dim: int,
            layer_idx: int,
            is_global: bool,
            has_cross_attn: bool,
            qk_norm_cross: bool = False,
            norm_eps: float = 1e-5,
            dropout: float = 0.,
            attn_drop: float = 0.,
    ):
        super().__init__()
        self.is_global = is_global
        self.dropout = dropout
        self.attn_norm = nn.LayerNorm(dim, eps=norm_eps, bias=False) if layer_idx != 0 else nn.Identity()
        self.attn = RopeSelfAttention(dim, num_heads, attn_drop=attn_drop)
        if has_cross_attn:
            self.cross_attn_norm = nn.LayerNorm(dim, eps=norm_eps, bias=False)
            self.cross_attn = Attention(dim, num_heads, attn_drop=attn_drop, qk_norm=qk_norm_cross)
        else:
            self.cross_attn_norm = None
            self.cross_attn = None
        self.mlp_norm = nn.LayerNorm(dim, eps=norm_eps, bias=False)
        self.mlp = GeGLU(dim, ffn_dim, drop=dropout)

    def forward(self, x, rope, cross_kv=None, cross_mask=None, self_mask=None, is_causal=False, cache=None):
        x = x + F.dropout(
            self.attn(self.attn_norm(x), rope, attn_mask=self_mask, is_causal=is_causal, cache=cache),
            p=self.dropout, training=self.training,
        )
        if self.cross_attn is not None:
            x = x + F.dropout(
                self.cross_attn(self.cross_attn_norm(x), kv=cross_kv, attn_mask=cross_mask),
                p=self.dropout, training=self.training,
            )
        x = x + self.mlp(self.mlp_norm(x))
        return x


class EttinTextDecoder(nn.Module):
    def __init__(
            self,
            cfg: TextDecoderCfg,
            vocab_size: int,
            pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.grad_checkpointing = False

        pretrained_state = None
        if cfg.pretrained:
            cfg, pretrained_state = _load_hf_ettin_decoder(cfg)
        elif cfg.num_layers is None:
            # unresolved config w/o pretrained weights, architecture from the hub config (random init)
            cfg = _resolve_hf_ettin_cfg(cfg, _load_hf_config(cfg.name))
        self.cfg = cfg
        self.dim = cfg.dim
        self.max_length = cfg.max_length
        self.sliding_window = cfg.sliding_window
        head_dim = cfg.dim // cfg.num_heads

        self.embed_tokens = nn.Embedding(vocab_size, cfg.dim, padding_idx=pad_token_id)
        self.embed_norm = nn.LayerNorm(cfg.dim, eps=cfg.norm_eps, bias=False)
        cross_every = cfg.cross_attn_every  # <= 0 disables cross-attention (prefix / decoder-only use)
        self.has_cross_attn = cross_every > 0
        self.layers = nn.ModuleList([
            EttinDecoderLayer(
                cfg.dim,
                cfg.num_heads,
                cfg.ffn_dim,
                layer_idx=i,
                is_global=i % cfg.global_attn_every == 0,
                has_cross_attn=cross_every > 0 and i % cross_every == cross_every - 1,
                qk_norm_cross=cfg.qk_norm_cross,
                norm_eps=cfg.norm_eps,
                dropout=cfg.dropout,
                attn_drop=cfg.attn_dropout,
            )
            for i in range(cfg.num_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.dim, eps=cfg.norm_eps, bias=False)
        self.head_dense = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.head_norm = nn.LayerNorm(cfg.dim, eps=cfg.norm_eps, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self.init_weights()
        if pretrained_state is not None:
            self._load_pretrained(pretrained_state)

    def init_weights(self):
        std = 0.02
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
        for layer in self.layers:
            if layer.cross_attn is not None:
                nn.init.zeros_(layer.cross_attn.out_proj.weight)

    @torch.no_grad()
    def _load_pretrained(self, state_dict: Dict[str, torch.Tensor]):
        own = self.state_dict()
        loaded = {}
        for k, v in state_dict.items():
            if k not in own:
                continue
            if k in ('embed_tokens.weight', 'output_bias'):
                n = min(v.shape[0], own[k].shape[0])
                w = own[k].clone()
                w[:n] = v[:n]
                if own[k].shape[0] > v.shape[0]:
                    w[v.shape[0]:] = v.mean(0, keepdim=True)
                v = w
            if v.shape != own[k].shape:
                _logger.warning(f'Skipping {k}, shape mismatch {v.shape} != {own[k].shape}')
                continue
            loaded[k] = v
        missing, unexpected = self.load_state_dict(loaded, strict=False)
        if missing:
            _logger.info(f'Text decoder params not in pretrained weights (newly initialized): {missing}')

    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def no_weight_decay(self):
        return {'embed_tokens.weight', 'output_bias'}

    def prepare_cross(self, encoder_hidden_states: torch.Tensor, encoder_mask: Optional[torch.Tensor] = None):
        """ Encoder output -> (per-layer cross-attn K / V, cross-attn mask) w/ the configured backend. """
        states, mask = prepare_cross_attn(encoder_hidden_states, encoder_mask, self.cfg.cross_attn_backend)
        return self.project_cross_kv(states), mask

    def project_cross_kv(self, encoder_hidden_states: torch.Tensor) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        return [
            layer.cross_attn.project_kv(encoder_hidden_states) if layer.cross_attn is not None else None
            for layer in self.layers
        ]

    def _rope(self, position_ids: torch.Tensor, dtype):
        freqs = position_ids[..., None].float() * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[:, None], emb.sin()[:, None]  # [B, 1, T, D]

    def forward(
            self,
            input_ids: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_mask: Optional[torch.Tensor] = None,
            cross_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cache: Optional[List[Dict[str, torch.Tensor]]] = None,
            prefix_embeds: Optional[torch.Tensor] = None,
            prefix_mask: Optional[torch.Tensor] = None,
            prefix_len: Optional[int] = None,
            cross_mask: Optional[CrossAttnMask] = None,
            return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T] text token ids
            encoder_hidden_states / encoder_mask / cross_kv: cross-attention inputs (if cross-attn enabled)
            position_ids: [B, T(+P)] RoPE positions, default derived from attention_mask (or arange)
            attention_mask: [B, past + P + T] bool key validity over the full sequence
            cache: per-layer self-attention KV caches
            prefix_embeds: [B, P, C] prefix (e.g. image) embeddings, prepended to the text, bidirectional
                attention within the prefix (prefix-LM). Only valid w/o past cache (training or prefill).
            prefix_mask: [B, P] bool validity of prefix positions
            prefix_len: length of a prefix already in the cache (incremental decoding), so sliding window layers
                keep full visibility of the prefix

        Returns:
            logits [B, T, V] for the text positions only
        """
        B, T = input_ids.shape
        device = input_ids.device
        past_len = cache[0]['k'].shape[2] if cache is not None and 'k' in cache[0] else 0

        x = self.embed_norm(self.embed_tokens(input_ids))
        P = 0
        if prefix_embeds is not None:
            assert past_len == 0, 'Prefix embeddings can only be passed w/o past cache.'
            P = prefix_embeds.shape[1]
            x = torch.cat([self.embed_norm(prefix_embeds).to(x.dtype), x], dim=1)
            if attention_mask is None and prefix_mask is not None:
                attention_mask = torch.cat([prefix_mask.bool(), prefix_mask.new_ones(B, T, dtype=torch.bool)], dim=1)
        x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        L = P + T  # current (new) sequence length
        S = past_len + L  # total key length

        if position_ids is None:
            if attention_mask is not None:
                # contiguous positions over valid tokens (padding inside the prefix or left padded prompts)
                position_ids = (attention_mask[:, :S].long().cumsum(1) - 1).clamp(min=0)[:, past_len:]
            else:
                position_ids = torch.arange(past_len, S, device=device).unsqueeze(0).expand(B, -1)
        rope = self._rope(position_ids, None)

        if self.has_cross_attn:
            if cross_kv is None:
                assert encoder_hidden_states is not None
                # K / V projected inside each layer's forward (per layer FSDP units, checkpointed w/ the layer)
                states, cross_mask = prepare_cross_attn(
                    encoder_hidden_states, encoder_mask, self.cfg.cross_attn_backend)
                cross_kv = [states] * len(self.layers)
            elif cross_mask is None and encoder_mask is not None:
                cross_mask = CrossAttnMask('sdpa', encoder_mask.bool())
        else:
            cross_kv = [None] * len(self.layers)
            cross_mask = None

        use_flex = (
            self.cfg.self_attn_backend == 'flex' and has_flex and past_len == 0 and L > 1
            and (P > 0 or attention_mask is not None)
        )
        if use_flex:
            key_valid = attention_mask[:, :S].bool() if attention_mask is not None else \
                torch.ones(B, S, dtype=torch.bool, device=device)
            global_mask = build_prefix_lm_block_mask(key_valid, P)
            local_mask = build_prefix_lm_block_mask(key_valid, P, window=self.sliding_window)
        # self-attention masks (bool, True == attend), sequence index based like HF
        if not use_flex:
            q_idx = torch.arange(past_len, S, device=device)[:, None]
            k_idx = torch.arange(S, device=device)[None, :]
            allowed = k_idx <= q_idx
            if P > 0:
                allowed = allowed | ((q_idx < P) & (k_idx < P))  # bidirectional prefix
            local = allowed & (k_idx > q_idx - self.sliding_window) & (k_idx < q_idx + self.sliding_window)
            full_prefix = P if P > 0 else (prefix_len or 0)
            if full_prefix > 0:
                # text queries see the entire prefix in sliding window layers too (window applies within prefix / text)
                local = local | (allowed & (k_idx < full_prefix) & (q_idx >= full_prefix))
            if attention_mask is not None:
                key_valid = attention_mask[:, None, None, :S].bool()
                diag = k_idx == q_idx
                global_mask = (allowed[None, None] & key_valid) | diag[None, None]
                local_mask = (local[None, None] & key_valid) | diag[None, None]
            else:
                simple_causal = P == 0 and (past_len == 0 or L == 1)
                global_mask = None if simple_causal else allowed[None, None]
                local_mask = local[None, None]

        for i, layer in enumerate(self.layers):
            mask = global_mask if layer.is_global else local_mask
            is_causal = (not use_flex) and layer.is_global and mask is None and L > 1
            layer_cache = cache[i] if cache is not None else None
            if self.grad_checkpointing and self.training and layer_cache is None:
                x = checkpoint(layer, x, rope, cross_kv[i], cross_mask, mask, is_causal, None, use_reentrant=False)
            else:
                x = layer(x, rope, cross_kv[i], cross_mask, mask, is_causal, layer_cache)

        x = x[:, P:]
        x = self.final_norm(x)
        x = self.head_norm(F.gelu(self.head_dense(x)))
        if return_hidden:
            return x
        weight, bias = self.lm_head_params()
        return F.linear(x, weight, bias)

    def lm_head_params(self):
        """ Output projection (weight, bias) applied to `return_hidden` features. """
        return self.embed_tokens.weight, self.output_bias


def _load_hf_config(name: str) -> Dict[str, Any]:
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(name, 'config.json')) as f:
        return json.load(f)


def _resolve_hf_ettin_cfg(cfg: TextDecoderCfg, hf_cfg: Dict[str, Any]) -> TextDecoderCfg:
    assert hf_cfg.get('model_type') == 'modernbert-decoder', f"Unsupported model type {hf_cfg.get('model_type')}"
    assert hf_cfg.get('hidden_activation', 'gelu') == 'gelu' and hf_cfg.get('classifier_activation', 'gelu') == 'gelu'
    assert hf_cfg.get('global_rope_theta') == hf_cfg.get('local_rope_theta', hf_cfg.get('global_rope_theta')), \
        'Different global / local rope theta not supported.'
    num_layers = cfg.num_layers or hf_cfg['num_hidden_layers']
    assert num_layers <= hf_cfg['num_hidden_layers']
    return replace(
        cfg,
        dim=hf_cfg['hidden_size'],
        num_heads=hf_cfg['num_attention_heads'],
        ffn_dim=hf_cfg['intermediate_size'],
        num_layers=num_layers,
        norm_eps=hf_cfg.get('norm_eps', 1e-5),
        rope_theta=hf_cfg.get('global_rope_theta', 160000.),
        sliding_window=hf_cfg.get('local_attention', 128) // 2,
        global_attn_every=hf_cfg.get('global_attn_every_n_layers', 3),
        max_length=min(cfg.max_length, hf_cfg.get('max_position_embeddings', cfg.max_length)),
        dropout=0.,
        attn_dropout=hf_cfg.get('attention_dropout', 0.),
    )


def _load_hf_ettin_decoder(cfg: TextDecoderCfg) -> Tuple[TextDecoderCfg, Dict[str, torch.Tensor]]:
    from huggingface_hub import hf_hub_download, list_repo_files

    cfg = _resolve_hf_ettin_cfg(cfg, _load_hf_config(cfg.name))
    num_layers = cfg.num_layers
    files = list_repo_files(cfg.name)
    if 'model.safetensors' in files:
        from safetensors.torch import load_file
        state_dict = load_file(hf_hub_download(cfg.name, 'model.safetensors'))
    else:
        state_dict = torch.load(hf_hub_download(cfg.name, 'pytorch_model.bin'), map_location='cpu', weights_only=True)

    out = {}
    for k, v in state_dict.items():
        nk = None
        if k == 'model.embeddings.tok_embeddings.weight':
            nk = 'embed_tokens.weight'
        elif k == 'model.embeddings.norm.weight':
            nk = 'embed_norm.weight'
        elif k == 'model.final_norm.weight':
            nk = 'final_norm.weight'
        elif k == 'lm_head.dense.weight':
            nk = 'head_dense.weight'
        elif k == 'lm_head.norm.weight':
            nk = 'head_norm.weight'
        elif k == 'decoder.bias':
            nk = 'output_bias'
        elif k.startswith('model.layers.'):
            nk = k[len('model.'):]
            if int(nk.split('.')[1]) >= num_layers:
                nk = None
        if nk is not None:
            out[nk] = v
    return cfg, out
