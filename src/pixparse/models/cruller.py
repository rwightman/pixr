from dataclasses import replace
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .config import ModelCfg, TextDecoderCfg
from .image_encoder_timm import ImageEncoderTimm
from .text_decoder import TextDecoder
from .text_decoder_ettin import EttinTextDecoder


def create_text_decoder(cfg: TextDecoderCfg, vocab_size: int, pad_token_id: int):
    if cfg.type == 'bart':
        return TextDecoder(cfg, vocab_size=vocab_size, pad_token_id=pad_token_id)
    elif cfg.type == 'ettin':
        return EttinTextDecoder(cfg, vocab_size=vocab_size, pad_token_id=pad_token_id)
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
    ) -> torch.Tensor:
        """ Teacher forced forward, returns logits [B, T, V]. """
        encoder_output, encoder_mask = self.image_encoder(image_input)
        return self.text_decoder(
            text_input,
            encoder_hidden_states=encoder_output,
            encoder_mask=encoder_mask,
        )

    def prefill(self, image_input, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor):
        """ Encode image + (left padded) prompts, returns next token logits and decoding state. """
        encoder_output, encoder_mask = self.encode(image_input)
        cross_kv = self.text_decoder.project_cross_kv(encoder_output)
        cache = [{} for _ in self.text_decoder.layers]
        position_ids = (prompt_mask.long().cumsum(1) - 1).clamp(min=0)
        logits = self.text_decoder(
            prompt_ids,
            cross_kv=cross_kv,
            encoder_mask=encoder_mask,
            position_ids=position_ids,
            attention_mask=prompt_mask,
            cache=cache,
        )
        state = dict(
            cross_kv=cross_kv,
            encoder_mask=encoder_mask,
            cache=cache,
            attn_mask=prompt_mask,
            pos=position_ids[:, -1],
        )
        return logits[:, -1], state

    def step(self, token_ids: torch.Tensor, state: Dict[str, Any]):
        """ Decode one token per sample w/ cached state, returns next token logits and updated state. """
        state['pos'] = state['pos'] + 1
        state['attn_mask'] = torch.cat([state['attn_mask'], state['attn_mask'].new_ones(token_ids.shape)], dim=1)
        logits = self.text_decoder(
            token_ids,
            cross_kv=state['cross_kv'],
            encoder_mask=state['encoder_mask'],
            position_ids=state['pos'][:, None],
            attention_mask=state['attn_mask'],
            cache=state['cache'],
        )
        return logits[:, -1], state
