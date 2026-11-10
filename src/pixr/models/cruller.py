from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .config import ModelCfg, TextDecoderCfg
from .image_encoder_timm import ImageEncoderTimm
from .kv_cache import CrossMemory, DecodeState, LayerKVCache
from .text_decoder import TextDecoder
from .text_decoder_ettin import EttinTextDecoder
from .text_decoder_qwen3 import Qwen3TextDecoder
from .param_groups import block_modules, complete_layer_map, decoder_layer_map, timm_layer_map


def create_text_decoder(cfg: TextDecoderCfg, vocab_size: int, pad_token_id: int):
    if cfg.type == 'bart':
        return TextDecoder(cfg, vocab_size=vocab_size, pad_token_id=pad_token_id)
    elif cfg.type == 'ettin':
        return EttinTextDecoder(cfg, vocab_size=vocab_size, pad_token_id=pad_token_id)
    elif cfg.type == 'qwen3':
        return Qwen3TextDecoder(cfg, vocab_size=vocab_size, pad_token_id=pad_token_id)
    raise ValueError(f'Unknown text decoder type {cfg.type}')


class Cruller(nn.Module):
    """ Image encoder (timm) + autoregressive text decoder w/ cross-attention (Donut style). """

    def __init__(self, cfg: ModelCfg, vocab_size: int, pad_token_id: int = 1):
        super().__init__()
        self.text_decoder = create_text_decoder(cfg.text_decoder, vocab_size=vocab_size, pad_token_id=pad_token_id)
        self.image_encoder = ImageEncoderTimm(cfg.image_encoder, out_dim=self.text_decoder.dim)
        # resolved config, sufficient to rebuild the model architecture (w/o pretrained weights)
        self.cfg = replace(cfg, image_encoder=self.image_encoder.cfg, text_decoder=self.text_decoder.cfg)

    @property
    def max_text_length(self) -> int:
        return self.text_decoder.max_length

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.image_encoder.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)

    def fsdp_shard_modules(self) -> List[Tuple[str, nn.Module]]:
        """ (name, module) blocks wrapped individually for FSDP2 / composable activation checkpointing. """
        return (
            block_modules(self.image_encoder.trunk, 'image_encoder.trunk.')
            + block_modules(self.text_decoder, 'text_decoder.', 'layers')
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd = set()
        no_wd |= {'image_encoder.' + n for n in self.image_encoder.no_weight_decay()}
        no_wd |= {'text_decoder.' + n for n in self.text_decoder.no_weight_decay()}
        return no_wd

    def encode(
            self,
            image_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.image_encoder(image_input)

    def forward(
            self,
            image_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
            text_input: torch.Tensor,
            return_hidden: bool = False,
    ) -> torch.Tensor:
        """ Teacher forced forward, returns logits [B, T, V] (or pre-head features [B, T, D] w/ return_hidden). """
        encoder_output, encoder_mask = self.image_encoder(image_input)
        return self.text_decoder(
            text_input,
            encoder_hidden_states=encoder_output,
            encoder_mask=encoder_mask,
            return_hidden=return_hidden,
        )

    def lm_head_params(self):
        """ Output projection (weight, bias) applied to `forward(..., return_hidden=True)` features. """
        return self.text_decoder.lm_head_params()

    def param_layer_map(self):
        """ {param name: (component, layer id, num layers)} for per component LR scales / layer-wise LR decay. """
        layer_map = timm_layer_map(self.image_encoder.trunk, 'image_encoder.trunk.')
        layer_map.update(decoder_layer_map(self.text_decoder, 'text_decoder.'))
        return complete_layer_map(self, layer_map)

    def encode_image(self, image_input: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> CrossMemory:
        """ Image -> per image decoding memory (encoder states prepared for cross-attention). Computed once per
        image, shared by all prompts / beams decoded for it. """
        encoder_output, encoder_mask = self.encode(image_input)
        return self.text_decoder.prepare_cross(encoder_output, encoder_mask)

    def prefill(self, memory: CrossMemory, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor):
        """ Run the (left padded) prompts, returns next token logits [N, V] and the decoding state.

        N prompts may be a multiple of the memory's image batch (several prompts per image, image-major order).
        """
        cache = [LayerKVCache() for _ in self.text_decoder.layers]
        position_ids = (prompt_mask.long().cumsum(1) - 1).clamp(min=0)
        logits = self.text_decoder(
            prompt_ids,
            cross_memory=memory,
            position_ids=position_ids,
            attention_mask=prompt_mask,
            cache=cache,
        )
        state = DecodeState(cache=cache, attn_mask=prompt_mask, pos=position_ids[:, -1], memory=memory)
        return logits[:, -1], state

    def step(self, token_ids: torch.Tensor, state: DecodeState):
        """ Decode one token per hypothesis w/ cached state, returns next token logits and the updated state. """
        state.pos = state.pos + 1
        state.attn_mask = torch.cat([state.attn_mask, state.attn_mask.new_ones(token_ids.shape)], dim=1)
        logits = self.text_decoder(
            token_ids,
            cross_memory=state.memory,
            position_ids=state.pos[:, None],
            attention_mask=state.attn_mask,
            cache=state.cache,
        )
        return logits[:, -1], state
