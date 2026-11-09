""" Cronut: native (encoder-free) decoder-only image + text transformer.

Placeholder. Planned design (GenLIP style, for document reading / parsing):
* a single transformer trunk over both modalities, NaFlex patches embedded w/ an image specific projection
  (optionally modality specific norms / MLPs), text tokens w/ the (pretrained) LM embedding
* 2D RoPE (or MRoPE) positions for image patches, 1D for text
* prefix-LM attention: bidirectional over [image ; prompt], causal over the answer / transcription
* same tasks, data pipelines, loss masking, and prefill() / step() generation interface as Cruller and Fritter
"""
import torch.nn as nn

from .config import ModelCfg


class Cronut(nn.Module):
    def __init__(self, cfg: ModelCfg, vocab_size: int, pad_token_id: int = 0):
        super().__init__()
        raise NotImplementedError('Cronut (native decoder-only image + text model) is not implemented yet.')
