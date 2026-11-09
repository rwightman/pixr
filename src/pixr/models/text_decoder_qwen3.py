""" Qwen3 style decoder-only LM (prefix-LM capable)

Pre-norm transformer w/ RMSNorm, GQA attention w/ per-head q/k RMSNorm, SwiGLU MLP, RoPE. Weights are loaded from
Hugging Face Qwen3 checkpoints (e.g. Qwen/Qwen3-0.6B-Base) by name.

Used as a prefix LM (Fritter), or w/ optional cross-attention (Cruller): a pre-norm cross-attention sub-block w/ zero
init output projection is inserted after self-attention in every `cross_attn_every` layers (Flamingo / Llama 3.2
Vision style), so the decoder starts as the pretrained LM.
"""
import json
import logging
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import TextDecoderCfg
from .text_decoder import Attention
from .text_decoder_ettin import apply_rope

_logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class GqaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)

    def forward(self, x, rope, attn_mask=None, is_causal=False, cache=None):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = rope
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if cache is not None:
            if 'k' in cache:
                k = torch.cat([cache['k'], k], dim=2)
                v = torch.cat([cache['v'], v], dim=2)
            cache['k'], cache['v'] = k, v
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal and attn_mask is None,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        return self.o_proj(x.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Layer(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, head_dim, ffn_dim, eps=1e-6, has_cross_attn=False,
                 qk_norm_cross=False):
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps=eps)
        self.self_attn = GqaAttention(dim, num_heads, num_kv_heads, head_dim, eps=eps)
        if has_cross_attn:
            self.cross_attn_norm = RMSNorm(dim, eps=eps)
            self.cross_attn = Attention(dim, num_heads, qk_norm=qk_norm_cross)
        else:
            self.cross_attn_norm = None
            self.cross_attn = None
        self.post_attention_layernorm = RMSNorm(dim, eps=eps)
        self.mlp = SwiGLU(dim, ffn_dim)

    def forward(self, x, rope, mask=None, is_causal=False, cache=None, cross_kv=None, cross_mask=None):
        x = x + self.self_attn(self.input_layernorm(x), rope, attn_mask=mask, is_causal=is_causal, cache=cache)
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.cross_attn_norm(x), kv=cross_kv, attn_mask=cross_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3TextDecoder(nn.Module):
    def __init__(self, cfg: TextDecoderCfg, vocab_size: int, pad_token_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.grad_checkpointing = False

        pretrained_state = None
        if cfg.pretrained:
            cfg, pretrained_state = _load_hf_qwen3(cfg)
        self.cfg = cfg
        self.dim = cfg.dim
        self.max_length = cfg.max_length
        cross_every = cfg.cross_attn_every
        self.has_cross_attn = cross_every > 0
        head_dim = cfg.head_dim or cfg.dim // cfg.num_heads
        num_kv_heads = cfg.num_kv_heads or cfg.num_heads

        self.embed_tokens = nn.Embedding(vocab_size, cfg.dim)
        self.layers = nn.ModuleList([
            Qwen3Layer(
                cfg.dim, cfg.num_heads, num_kv_heads, head_dim, cfg.ffn_dim,
                eps=cfg.norm_eps,
                has_cross_attn=cross_every > 0 and i % cross_every == cross_every - 1,
                qk_norm_cross=cfg.qk_norm_cross,
            )
            for i in range(cfg.num_layers)
        ])
        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self.init_weights()
        if pretrained_state is not None:
            self._load_pretrained(pretrained_state)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
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
            if k == 'embed_tokens.weight' and v.shape != own[k].shape:
                n = min(v.shape[0], own[k].shape[0])
                w = own[k].clone()
                w[:n] = v[:n]
                if own[k].shape[0] > v.shape[0]:
                    w[v.shape[0]:] = v[:n].mean(0, keepdim=True)
                v = w
            if v.shape != own[k].shape:
                _logger.warning(f'Skipping {k}, shape mismatch {v.shape} != {own[k].shape}')
                continue
            loaded[k] = v
        missing, _ = self.load_state_dict(loaded, strict=False)
        if missing:
            _logger.info(f'Text decoder params not in pretrained weights (newly initialized): {missing}')

    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def no_weight_decay(self):
        return {'embed_tokens.weight'}

    def project_cross_kv(self, encoder_hidden_states: torch.Tensor):
        return [
            layer.cross_attn.project_kv(encoder_hidden_states) if layer.cross_attn is not None else None
            for layer in self.layers
        ]

    def _rope(self, position_ids: torch.Tensor):
        freqs = position_ids[..., None].float() * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[:, None], emb.sin()[:, None]

    def forward(
            self,
            input_ids: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_mask: Optional[torch.Tensor] = None,
            cross_kv=None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cache: Optional[List[Dict[str, torch.Tensor]]] = None,
            prefix_embeds: Optional[torch.Tensor] = None,
            prefix_mask: Optional[torch.Tensor] = None,
            prefix_len: Optional[int] = None,
    ) -> torch.Tensor:
        """ Same interface as EttinTextDecoder (prefix-LM and / or cross-attn), returns text position logits. """
        B, T = input_ids.shape
        device = input_ids.device
        past_len = cache[0]['k'].shape[2] if cache is not None and 'k' in cache[0] else 0

        x = self.embed_tokens(input_ids)
        P = 0
        if prefix_embeds is not None:
            assert past_len == 0, 'Prefix embeddings can only be passed w/o past cache.'
            P = prefix_embeds.shape[1]
            x = torch.cat([prefix_embeds.to(x.dtype), x], dim=1)
            if attention_mask is None and prefix_mask is not None:
                attention_mask = torch.cat([prefix_mask.bool(), prefix_mask.new_ones(B, T, dtype=torch.bool)], dim=1)
        L = P + T
        S = past_len + L

        if position_ids is None:
            if attention_mask is not None:
                position_ids = (attention_mask[:, :S].long().cumsum(1) - 1).clamp(min=0)[:, past_len:]
            else:
                position_ids = torch.arange(past_len, S, device=device).unsqueeze(0).expand(B, -1)
        rope = self._rope(position_ids)

        if self.has_cross_attn:
            if cross_kv is None:
                assert encoder_hidden_states is not None
                cross_kv = self.project_cross_kv(encoder_hidden_states)
            cross_mask = encoder_mask[:, None, None, :] if encoder_mask is not None else None
        else:
            cross_kv = [None] * len(self.layers)
            cross_mask = None

        q_idx = torch.arange(past_len, S, device=device)[:, None]
        k_idx = torch.arange(S, device=device)[None, :]
        allowed = k_idx <= q_idx
        if P > 0:
            allowed = allowed | ((q_idx < P) & (k_idx < P))  # bidirectional prefix
        if attention_mask is not None:
            key_valid = attention_mask[:, None, None, :S].bool()
            mask = (allowed[None, None] & key_valid) | (k_idx == q_idx)[None, None]
        else:
            simple_causal = P == 0 and (past_len == 0 or L == 1)
            mask = None if simple_causal else allowed[None, None]
        is_causal = mask is None and L > 1

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            if self.grad_checkpointing and self.training and layer_cache is None:
                x = checkpoint(layer, x, rope, mask, is_causal, None, cross_kv[i], cross_mask, use_reentrant=False)
            else:
                x = layer(x, rope, mask, is_causal, layer_cache, cross_kv[i], cross_mask)

        x = self.norm(x[:, P:])
        return F.linear(x, self.embed_tokens.weight)


def _load_hf_qwen3(cfg: TextDecoderCfg) -> Tuple[TextDecoderCfg, Dict[str, torch.Tensor]]:
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.torch import load_file

    with open(hf_hub_download(cfg.name, 'config.json')) as f:
        hf_cfg = json.load(f)
    assert hf_cfg.get('model_type') == 'qwen3', f"Unsupported model type {hf_cfg.get('model_type')}"
    assert hf_cfg.get('tie_word_embeddings', False), 'Untied output embeddings not supported yet.'
    assert not hf_cfg.get('use_sliding_window', False) and not hf_cfg.get('rope_scaling')
    files = [f for f in list_repo_files(cfg.name) if f.endswith('.safetensors')]
    state_dict = {}
    for f in files:
        state_dict.update(load_file(hf_hub_download(cfg.name, f)))

    num_layers = cfg.num_layers or hf_cfg['num_hidden_layers']
    assert num_layers <= hf_cfg['num_hidden_layers']
    cfg = replace(
        cfg,
        dim=hf_cfg['hidden_size'],
        num_heads=hf_cfg['num_attention_heads'],
        num_kv_heads=hf_cfg['num_key_value_heads'],
        head_dim=hf_cfg.get('head_dim') or hf_cfg['hidden_size'] // hf_cfg['num_attention_heads'],
        ffn_dim=hf_cfg['intermediate_size'],
        num_layers=num_layers,
        norm_eps=hf_cfg.get('rms_norm_eps', 1e-6),
        rope_theta=float(hf_cfg.get('rope_theta', 1e6)),
        max_length=min(cfg.max_length, hf_cfg.get('max_position_embeddings', cfg.max_length)),
        dropout=0.,
        attn_dropout=0.,
    )
    out = {}
    for k, v in state_dict.items():
        if not k.startswith('model.'):
            continue  # lm_head tied to embed_tokens
        nk = k[len('model.'):]
        if nk.startswith('layers.') and int(nk.split('.')[1]) >= num_layers:
            continue
        out[nk] = v
    return cfg, out
