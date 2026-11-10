""" Cronut: native decoder-only image + text transformer (GenLIP style) for document reading / parsing.

A single transformer trunk runs over [image patches ; text tokens] w/ prefix-LM attention: image patches attend
bidirectionally among themselves, text attends causally to text and to all image patches, image never attends to
text. Follows the open_clip NaFlex GenLIP modelling:

* NaFlex linear patch embedding (no absolute position embedding), optional input / post-projection norms
* interleaved 3-axis MRoPE: image patches at (t=0, y, x), text at a running 1D index after the image extent
* gated attention (sigmoid output gate fused into q_proj), optional qk-norm, SwiGLU FFN, LayerScale, DropPath
* untied LM head, text embedding dim decoupled from trunk width via in / out projections

Attention backends: 'sdpa' (dense prefix-LM bool mask) or 'flex' (FlexAttention prefix-LM block mask).
Same training forward and prefill() / step() generation interface as Cruller and Fritter.
"""
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.layers import DropPath

from .attention import build_prefix_lm_block_mask, self_attention, has_flex
from .config import CronutCfg, ImageEncoderCfg, ModelCfg
from .param_groups import decoder_layer_map

_ACT = {'silu': F.silu, 'gelu': F.gelu}


def _norm_layer(norm_type: str, eps: float):
    if norm_type == 'rmsnorm':
        return lambda d: nn.RMSNorm(d, eps=eps)
    assert norm_type == 'layernorm', f'Unknown norm_type {norm_type}'
    return lambda d: nn.LayerNorm(d, eps=eps)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class MRoPE(nn.Module):
    """ 3-axis (T / H / W) interleaved multimodal rotary embedding. """

    def __init__(self, head_dim: int, mrope_section: Tuple[int, int, int], theta: float, interleaved: bool = True):
        super().__init__()
        assert sum(mrope_section) == head_dim // 2, \
            f'sum(mrope_section)={sum(mrope_section)} must equal head_dim // 2 = {head_dim // 2}'
        self.mrope_section = tuple(mrope_section)
        self.interleaved = interleaved
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """ position_ids [3, B, S] -> cos / sin [B, 1, S, head_dim] """
        freqs = position_ids[..., None].float() * self.inv_freq.float()  # 3, B, S, head_dim // 2
        if self.interleaved:
            f = freqs[0].clone()
            for axis in (1, 2):
                length = self.mrope_section[axis] * 3
                f[..., axis:length:3] = freqs[axis, ..., axis:length:3]
            freqs = f
        else:
            freqs = freqs[0]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[:, None].to(dtype), emb.sin()[:, None].to(dtype)


class CronutAttention(nn.Module):
    def __init__(self, cfg: CronutCfg, norm_layer):
        super().__init__()
        self.width = cfg.width
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.width // cfg.num_heads
        self.gated = cfg.gated_attention
        bias = cfg.attention_bias
        self.q_proj = nn.Linear(cfg.width, cfg.width * (2 if self.gated else 1), bias=bias)
        self.k_proj = nn.Linear(cfg.width, cfg.width, bias=bias)
        self.v_proj = nn.Linear(cfg.width, cfg.width, bias=bias)
        self.out_proj = nn.Linear(cfg.width, cfg.width, bias=bias)
        self.q_norm = norm_layer(self.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if cfg.qk_norm else nn.Identity()

    def forward(self, x, cos, sin, attn_mask=None, is_causal=False, cache=None):
        B, S, _ = x.shape
        if self.gated:
            q, gate = self.q_proj(x).chunk(2, dim=-1)
        else:
            q, gate = self.q_proj(x), None
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q = (q * cos + rotate_half(q) * sin).to(v.dtype)
        k = (k * cos + rotate_half(k) * sin).to(v.dtype)
        if cache is not None:
            if 'k' in cache:
                k = torch.cat([cache['k'], k], dim=2)
                v = torch.cat([cache['v'], v], dim=2)
            cache['k'], cache['v'] = k, v
        out = self_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal).transpose(1, 2)  # B, S, H, D
        if gate is not None:
            out = out * torch.sigmoid(gate.view(B, S, self.num_heads, self.head_dim))
        return self.out_proj(out.reshape(B, S, self.width))


class CronutMlp(nn.Module):
    def __init__(self, cfg: CronutCfg):
        super().__init__()
        self.act = _ACT[cfg.hidden_act]
        self.swiglu = cfg.use_swiglu_ffn
        self.fc1 = nn.Linear(cfg.width, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.gate_fc = nn.Linear(cfg.width, cfg.intermediate_size, bias=cfg.mlp_bias) if self.swiglu else None
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.width, bias=cfg.mlp_bias)

    def forward(self, x):
        if self.swiglu:
            return self.fc2(self.act(self.gate_fc(x)) * self.fc1(x))
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class CronutBlock(nn.Module):
    def __init__(self, cfg: CronutCfg, norm_layer, drop_path: float = 0.):
        super().__init__()
        self.norm1 = norm_layer(cfg.width)
        self.attn = CronutAttention(cfg, norm_layer)
        self.norm2 = norm_layer(cfg.width)
        self.mlp = CronutMlp(cfg)
        use_ls = cfg.ls_init_value is not None and cfg.ls_init_value > 0
        self.ls1 = LayerScale(cfg.width, cfg.ls_init_value) if use_ls else nn.Identity()
        self.ls2 = LayerScale(cfg.width, cfg.ls_init_value) if use_ls else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, cos, sin, attn_mask=None, is_causal=False, cache=None):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), cos, sin, attn_mask, is_causal, cache)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class CronutPatchEmbed(nn.Module):
    """ Linear embedding of pre-patchified (NaFlex) inputs. Exposes the image input info used by tasks. """

    def __init__(self, enc_cfg: ImageEncoderCfg, cfg: CronutCfg, norm_layer):
        super().__init__()
        in_chans = 1 if enc_cfg.image_fmt == 'L' else 3
        patch_dim = cfg.patch_size * cfg.patch_size * in_chans
        self.norm_input = nn.LayerNorm(patch_dim) if cfg.patch_input_norm else nn.Identity()
        self.proj = nn.Linear(patch_dim, cfg.width, bias=True)
        self.norm_pre = norm_layer(cfg.width) if cfg.patch_pre_norm else nn.Identity()
        mean = tuple(enc_cfg.image_mean or (0.5,) * in_chans)
        std = tuple(enc_cfg.image_std or (0.5,) * in_chans)
        self.cfg = replace(enc_cfg, image_mean=mean, image_std=std, pretrained=False)
        self.num_prefix_tokens = 0
        self.input_info = dict(
            image_fmt=enc_cfg.image_fmt,
            image_size=None,
            max_seq_len=enc_cfg.max_seq_len,
            patch_size=cfg.patch_size,
            mean=mean,
            std=std,
            naflex=True,
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.norm_pre(self.proj(self.norm_input(patches)))


class Cronut(nn.Module):
    # vocab sized params, rows expanded when loading checkpoints into a model w/ added tokens
    vocab_param_names = ('text_embed.weight', 'lm_head.weight')

    def __init__(self, cfg: ModelCfg, vocab_size: int, pad_token_id: int = 0):
        super().__init__()
        c = cfg.cronut or CronutCfg()
        assert c.attn_backend in ('sdpa', 'flex'), f'Unknown Cronut attn_backend {c.attn_backend}'
        norm_layer = _norm_layer(c.norm_type, c.layer_norm_eps)
        self.c = c
        self.pad_token_id = pad_token_id
        self.max_length = cfg.text_decoder.max_length
        self.grad_checkpointing = False

        self.image_encoder = CronutPatchEmbed(cfg.image_encoder, c, norm_layer)
        self.text_embed = nn.Embedding(vocab_size, c.text_embed_dim, padding_idx=pad_token_id)
        self.in_proj = nn.Linear(c.text_embed_dim, c.width) if c.text_embed_dim != c.width else nn.Identity()
        self.text_norm_pre = norm_layer(c.width) if c.text_pre_norm else nn.Identity()
        self.rotary = MRoPE(c.width // c.num_heads, c.mrope_section, c.rope_theta, c.mrope_interleaved)
        dpr = [x.item() for x in torch.linspace(0, c.drop_path_rate, c.depth)]
        self.layers = nn.ModuleList([CronutBlock(c, norm_layer, drop_path=dpr[i]) for i in range(c.depth)])
        self.ln_post = norm_layer(c.width)
        self.out_proj = nn.Linear(c.width, c.text_embed_dim) if c.text_embed_dim != c.width else nn.Identity()
        self.lm_head = nn.Linear(c.text_embed_dim, vocab_size, bias=False)

        self.init_weights()
        self.cfg = replace(cfg, image_encoder=self.image_encoder.cfg, cronut=c)

    @torch.no_grad()
    def init_weights(self, std: float = 0.02):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=std)
                if m.padding_idx is not None:
                    m.weight[m.padding_idx].zero_()
        for m in self.modules():
            if isinstance(m, CronutAttention):
                for lin in (m.q_proj, m.k_proj, m.v_proj, m.out_proj):
                    nn.init.xavier_uniform_(lin.weight)
            elif isinstance(m, CronutMlp):
                for lin in (m.fc1, m.gate_fc, m.fc2):
                    if lin is not None:
                        nn.init.xavier_uniform_(lin.weight)

    @property
    def max_text_length(self) -> int:
        return self.max_length

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'text_embed.weight'}

    def embed_text(self, ids: torch.Tensor) -> torch.Tensor:
        return self.text_norm_pre(self.in_proj(self.text_embed(ids)))

    def _image_positions(self, image_input):
        coord, valid = image_input['patch_coord'], image_input['patch_valid'].bool()
        B, N, _ = coord.shape
        pos = coord.new_zeros((3, B, N), dtype=torch.long)
        pos[1] = coord[..., 0].long()
        pos[2] = coord[..., 1].long()
        h = torch.where(valid, pos[1], torch.zeros_like(pos[1]))
        w = torch.where(valid, pos[2], torch.zeros_like(pos[2]))
        text_start = torch.maximum(h.amax(1), w.amax(1)) + 1  # [B]
        return pos, text_start

    def _attn_mask(self, key_valid: torch.Tensor, prefix_len: int, x: torch.Tensor):
        if self.c.attn_backend == 'flex' and has_flex:
            return build_prefix_lm_block_mask(key_valid, prefix_len)
        S = key_valid.shape[1]
        idx = torch.arange(S, device=x.device)
        allowed = (idx[None, :] <= idx[:, None]) | ((idx[:, None] < prefix_len) & (idx[None, :] < prefix_len))
        mask = allowed[None] & key_valid[:, None, :]
        mask = mask | torch.eye(S, dtype=torch.bool, device=x.device)[None]
        return mask[:, None]

    def _run(self, x, cos, sin, attn_mask, cache=None):
        for i, blk in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            if self.grad_checkpointing and self.training and layer_cache is None:
                x = checkpoint(blk, x, cos, sin, attn_mask, False, None, use_reentrant=False)
            else:
                x = blk(x, cos, sin, attn_mask, False, layer_cache)
        return self.ln_post(x)

    def _logits(self, x):
        return self.lm_head(self.out_proj(x))

    def lm_head_params(self):
        """ Output projection (weight, bias) applied to `forward(..., return_hidden=True)` features. """
        return self.lm_head.weight, self.lm_head.bias

    def param_layer_map(self):
        """ {param name: (component, layer id, num layers)} for per component LR scales / layer-wise LR decay.

        The patch embed is the 'encoder' component, the shared trunk (+ text embed / head) the 'decoder'.
        """
        layer_map = decoder_layer_map(self, '', bridge=False, bottom=('text_embed', 'in_proj', 'text_norm_pre'))
        for n, _ in self.image_encoder.named_parameters():
            layer_map['image_encoder.' + n] = ('encoder', 0, 1)
        return layer_map

    def _prefix_forward(self, image_input, text_ids, text_mask, cache=None, return_hidden=False):
        """ Run [image ; text] through the trunk, returns text position logits and the final text positions. """
        assert isinstance(image_input, dict), 'Cronut requires NaFlex (patch dict) image input.'
        img = self.image_encoder(image_input['patches'])
        patch_valid = image_input['patch_valid'].bool()
        img_pos, text_start = self._image_positions(image_input)
        txt = self.embed_text(text_ids)
        x = torch.cat([img, txt.to(img.dtype)], dim=1)
        P = img.shape[1]
        txt_pos = text_start[:, None] + (text_mask.long().cumsum(1) - 1).clamp(min=0)  # B, T
        pos = torch.cat([img_pos, txt_pos[None].expand(3, -1, -1)], dim=2)
        cos, sin = self.rotary(pos, x.dtype)
        key_valid = torch.cat([patch_valid, text_mask.bool()], dim=1)
        x = self._run(x, cos, sin, self._attn_mask(key_valid, P, x), cache=cache)
        x = x[:, P:]
        x = self.out_proj(x) if return_hidden else self._logits(x)
        return x, key_valid, txt_pos[:, -1]

    def forward(
            self,
            image_input: Dict[str, torch.Tensor],
            text_input: torch.Tensor,
            return_hidden: bool = False,
    ) -> torch.Tensor:
        """ Teacher forced forward, returns text logits [B, T, V] (or pre-head features w/ return_hidden). """
        text_mask = torch.ones_like(text_input, dtype=torch.bool)  # right padded text, causal attention
        out, _, _ = self._prefix_forward(image_input, text_input, text_mask, return_hidden=return_hidden)
        return out

    def prefill(self, image_input, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor):
        cache = [{} for _ in self.layers]
        logits, key_valid, last_pos = self._prefix_forward(image_input, prompt_ids, prompt_mask, cache=cache)
        state = dict(cache=cache, key_valid=key_valid, pos=last_pos)
        return logits[:, -1], state

    def step(self, token_ids: torch.Tensor, state: Dict[str, Any]):
        state['pos'] = state['pos'] + 1
        state['key_valid'] = torch.cat([state['key_valid'], state['key_valid'].new_ones(token_ids.shape)], dim=1)
        x = self.embed_text(token_ids)
        pos = state['pos'][None, :, None].expand(3, -1, 1)
        cos, sin = self.rotary(pos, x.dtype)
        attn_mask = state['key_valid'][:, None, None, :]
        x = self._run(x, cos, sin, attn_mask, cache=state['cache'])
        return self._logits(x)[:, -1], state
