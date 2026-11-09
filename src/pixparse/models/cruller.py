from dataclasses import replace
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .config import ModelCfg
from .image_encoder_timm import ImageEncoderTimm
from .config import TextDecoderCfg
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
