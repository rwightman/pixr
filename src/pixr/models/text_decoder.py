""" BART style transformer text decoder w/ cross-attention

A self-contained implementation (weights loaded from Hugging Face BART checkpoints by name) so the decoder
does not depend on transformers model internals. Supports:
 * cross-attention key padding masks (variable length NaFlex encoder sequences)
 * optional qk-norm in cross-attention
 * KV caching for generation, per sample position ids (left padded prompts)
 * F.scaled_dot_product_attention
"""
import logging
import math
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import TextDecoderCfg

_logger = logging.getLogger(__name__)

# BART learned position embeddings are offset by 2 (legacy fairseq padding_idx handling)
_POS_OFFSET = 2


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            attn_drop: float = 0.,
            qk_norm: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

    def _split(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def project_kv(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.k_norm(self._split(self.k_proj(x)))
        v = self._split(self.v_proj(x))
        return k, v

    def forward(
            self,
            x: torch.Tensor,
            kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attn_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
    ):
        B, N, C = x.shape
        q = self.q_norm(self._split(self.q_proj(x)))
        if kv is None:
            kv = self.project_kv(x)
        k, v = kv
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.,
            is_causal=is_causal and attn_mask is None,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)


class DecoderLayer(nn.Module):
    """ Post-norm decoder layer (BART layout & parameter names). """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            ffn_dim: int,
            dropout: float = 0.1,
            attn_drop: float = 0.,
            act_drop: float = 0.,
            qk_norm_cross: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.act_drop = act_drop
        self.self_attn = Attention(dim, num_heads, attn_drop=attn_drop)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.encoder_attn = Attention(dim, num_heads, attn_drop=attn_drop, qk_norm=qk_norm_cross)
        self.encoder_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(
            self,
            x: torch.Tensor,
            cross_kv: Tuple[torch.Tensor, torch.Tensor],
            cross_mask: Optional[torch.Tensor] = None,
            self_mask: Optional[torch.Tensor] = None,
            cache: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # self-attention
        residual = x
        if cache is not None:
            k, v = self.self_attn.project_kv(x)
            if 'k' in cache:
                k = torch.cat([cache['k'], k], dim=2)
                v = torch.cat([cache['v'], v], dim=2)
            cache['k'], cache['v'] = k, v
            x = self.self_attn(x, kv=(k, v), attn_mask=self_mask, is_causal=self_mask is None and x.shape[1] > 1)
        else:
            x = self.self_attn(x, attn_mask=self_mask, is_causal=self_mask is None)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.self_attn_layer_norm(residual + x)

        # cross-attention
        residual = x
        x = self.encoder_attn(x, kv=cross_kv, attn_mask=cross_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.encoder_attn_layer_norm(residual + x)

        # mlp
        residual = x
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self.act_drop, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_layer_norm(residual + x)
        return x


class TextDecoder(nn.Module):
    def __init__(
            self,
            cfg: TextDecoderCfg,
            vocab_size: int,
            pad_token_id: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_length = cfg.max_length
        self.grad_checkpointing = False

        pretrained_state = None
        if cfg.pretrained:
            cfg, pretrained_state = _load_hf_bart_decoder(cfg)
        if cfg.scale_embedding is None:
            cfg = replace(cfg, scale_embedding=False)
        self.cfg = cfg  # resolved config (arch from pretrained) so the model can be rebuilt w/o pretrained weights
        self.dim = cfg.dim
        scale_embedding = cfg.scale_embedding
        self.embed_scale = math.sqrt(cfg.dim) if scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(vocab_size, cfg.dim, padding_idx=pad_token_id)
        self.embed_positions = nn.Embedding(cfg.max_length + _POS_OFFSET, cfg.dim)
        self.layernorm_embedding = nn.LayerNorm(cfg.dim)
        self.layers = nn.ModuleList([
            DecoderLayer(
                cfg.dim,
                cfg.num_heads,
                cfg.ffn_dim,
                dropout=cfg.dropout,
                attn_drop=cfg.attn_dropout,
                act_drop=cfg.act_dropout,
                qk_norm_cross=cfg.qk_norm_cross,
            )
            for _ in range(cfg.num_layers)
        ])
        self.dropout = cfg.dropout
        self.init_weights()
        if pretrained_state is not None:
            self._load_pretrained(pretrained_state)

    def init_weights(self):
        std = 0.02
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=std)
        with torch.no_grad():
            self.embed_tokens.weight[self.pad_token_id].zero_()
        if self.cfg.reinit_cross_attn:
            for layer in self.layers:
                nn.init.zeros_(layer.encoder_attn.out_proj.weight)

    @torch.no_grad()
    def _load_pretrained(self, state_dict: Dict[str, torch.Tensor]):
        own = self.state_dict()
        loaded = {}
        for k, v in state_dict.items():
            if k not in own:
                continue
            if self.cfg.reinit_cross_attn and '.encoder_attn.' in k:
                continue
            if k == 'embed_tokens.weight':
                n = min(v.shape[0], own[k].shape[0])
                w = own[k].clone()
                w[:n] = v[:n]
                if own[k].shape[0] > v.shape[0]:
                    # init added token rows w/ the mean of the pretrained embeddings
                    w[v.shape[0]:] = v.mean(0, keepdim=True)
                v = w
            elif k == 'embed_positions.weight' and v.shape != own[k].shape:
                v = resize_pos_embed(v, own[k].shape[0])
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
        return {'embed_tokens.weight', 'embed_positions.weight'}

    def resize_token_embeddings(self, new_vocab_size: int):
        old = self.embed_tokens.weight.data
        if new_vocab_size == old.shape[0]:
            return
        new = nn.Embedding(new_vocab_size, self.dim, padding_idx=self.pad_token_id).to(old.device, old.dtype)
        with torch.no_grad():
            n = min(old.shape[0], new_vocab_size)
            new.weight[:n] = old[:n]
            if new_vocab_size > old.shape[0]:
                new.weight[old.shape[0]:] = old.mean(0, keepdim=True)
        self.embed_tokens = new
        self.vocab_size = new_vocab_size

    def project_cross_kv(self, encoder_hidden_states: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [layer.encoder_attn.project_kv(encoder_hidden_states) for layer in self.layers]

    def forward(
            self,
            input_ids: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_mask: Optional[torch.Tensor] = None,
            cross_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cache: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T] token ids
            encoder_hidden_states: [B, S, C] encoder output (or pass pre-projected `cross_kv`)
            encoder_mask: [B, S] bool, True for valid encoder tokens
            cross_kv: per-layer pre-projected cross-attention keys & values (generation)
            position_ids: [B, T] position ids, default is arange(past_len, past_len + T)
            attention_mask: [B, past_len + T] bool key validity mask for self-attention (left padded prompts)
            cache: per-layer dicts for self-attention KV caching (generation)

        Returns:
            logits [B, T, vocab_size]
        """
        B, T = input_ids.shape
        past_len = cache[0]['k'].shape[2] if cache is not None and 'k' in cache[0] else 0
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(0)

        x = self.embed_tokens(input_ids) * self.embed_scale
        x = x + self.embed_positions(position_ids + _POS_OFFSET)
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if cross_kv is None:
            assert encoder_hidden_states is not None
            cross_kv = self.project_cross_kv(encoder_hidden_states)

        cross_mask = None
        if encoder_mask is not None:
            cross_mask = encoder_mask[:, None, None, :]  # [B, 1, 1, S]

        self_mask = None
        if attention_mask is not None or (past_len > 0 and T > 1):
            # explicit mask needed (padding, or multi-token step w/ cache), causal w/ offset + key padding
            S = past_len + T
            causal = torch.ones(T, S, dtype=torch.bool, device=input_ids.device).tril(diagonal=past_len)
            self_mask = causal[None, None]
            if attention_mask is not None:
                self_mask = self_mask & attention_mask[:, None, None, :S].bool()
                # always attend to self so padded query rows are never fully masked (avoids NaN)
                diag = torch.zeros(T, S, dtype=torch.bool, device=input_ids.device)
                diag[torch.arange(T), past_len + torch.arange(T)] = True
                self_mask = self_mask | diag[None, None]

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            if self.grad_checkpointing and self.training and layer_cache is None:
                x = checkpoint(layer, x, cross_kv[i], cross_mask, self_mask, None, use_reentrant=False)
            else:
                x = layer(x, cross_kv[i], cross_mask=cross_mask, self_mask=self_mask, cache=layer_cache)

        logits = F.linear(x, self.embed_tokens.weight)  # tied output projection (as BartForCausalLM)
        return logits


def resize_pos_embed(weight: torch.Tensor, new_len: int) -> torch.Tensor:
    """ Resize learned position embeddings (incl. offset rows), truncate or linear interpolate. """
    if weight.shape[0] >= new_len:
        return weight[:new_len]
    offset = weight[:_POS_OFFSET]
    pos = weight[_POS_OFFSET:]
    pos = F.interpolate(
        pos.T.unsqueeze(0),
        size=new_len - _POS_OFFSET,
        mode='linear',
        align_corners=False,
    ).squeeze(0).T
    return torch.cat([offset, pos], dim=0)


def _load_hf_bart_decoder(cfg: TextDecoderCfg) -> Tuple[TextDecoderCfg, Dict[str, torch.Tensor]]:
    """ Load a Hugging Face BART (or mBART) decoder state dict + arch hparams. """
    import json
    from huggingface_hub import hf_hub_download, list_repo_files

    with open(hf_hub_download(cfg.name, 'config.json')) as f:
        hf_cfg = json.load(f)
    files = list_repo_files(cfg.name)
    if 'model.safetensors' in files:
        from safetensors.torch import load_file
        state_dict = load_file(hf_hub_download(cfg.name, 'model.safetensors'))
    else:
        state_dict = torch.load(hf_hub_download(cfg.name, 'pytorch_model.bin'), map_location='cpu', weights_only=True)

    num_layers = cfg.num_layers or hf_cfg['decoder_layers']
    assert num_layers <= hf_cfg['decoder_layers'], \
        f"num_layers ({num_layers}) > pretrained decoder layers ({hf_cfg['decoder_layers']})"
    assert hf_cfg.get('activation_function', 'gelu') == 'gelu'
    assert not hf_cfg.get('normalize_before', False) and not hf_cfg.get('add_final_layer_norm', False), \
        'Only post-norm BART style decoders without final norm are currently supported.'
    scale_embedding = cfg.scale_embedding
    if scale_embedding is None:
        scale_embedding = hf_cfg.get('scale_embedding', False)
    cfg = replace(
        cfg,
        dim=hf_cfg['d_model'],
        num_heads=hf_cfg['decoder_attention_heads'],
        ffn_dim=hf_cfg['decoder_ffn_dim'],
        num_layers=num_layers,
        dropout=hf_cfg.get('dropout', cfg.dropout),
        attn_dropout=hf_cfg.get('attention_dropout', cfg.attn_dropout),
        act_dropout=hf_cfg.get('activation_dropout', cfg.act_dropout),
        scale_embedding=scale_embedding,
    )

    out = {}
    for k, v in state_dict.items():
        for prefix in ('model.decoder.', 'decoder.'):
            if k.startswith(prefix):
                nk = k[len(prefix):]
                if nk.startswith('layers.') and int(nk.split('.')[1]) >= num_layers:
                    break
                out[nk] = v
                break
    if 'embed_tokens.weight' not in out:
        for k in ('model.shared.weight', 'shared.weight'):
            if k in state_dict:
                out['embed_tokens.weight'] = state_dict[k]
                break
    return cfg, out
