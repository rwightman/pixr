""" Mutable state for incremental decoding.

Two lifetimes are kept apart:

* Image memory, per image: computed once, read-only afterwards, shared by every hypothesis decoded for that image
  (prompts, beams). `CrossMemory` (cross-attention models) or `PrefixMemory` (prefix-LM / decoder-only models).
* Sequence state, per hypothesis: `DecodeState` w/ the per-layer self-attention `LayerKVCache`, grows every step and
  is reordered (`DecodeState.reorder`) when hypotheses are selected / duplicated (beam search).

NOTE prefix-LM models (Fritter, Cronut) keep the image prefix K / V in the self-attention cache, so it is part of the
sequence state and copied for every hypothesis (beams x prefix length x layers) and on every reorder. Sharing it per
image as Cruller's cross-attention memory would need the prefix and text K / V attended separately (two segments).

All of these are plain classes (not dataclasses, dicts or lists) on purpose. FSDP2 mixed-precision input casting
rebuilds dataclasses / dicts / lists / tuples passed into a wrapped module's forward, updates made to them inside a
layer would be made to a copy and lost. Plain objects pass through by identity. Their tensors are therefore not cast
or moved by wrappers either, they keep the device / dtype of the projections that produced them.
"""
from typing import List, Optional

import torch


class LayerKVCache:
    """ Per-layer self-attention K / V of one batch of hypotheses, populated from the layer's projections. """
    __slots__ = ('k', 'v')

    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def reorder(self, parent_idx: torch.Tensor):
        """ Select (and possibly duplicate) hypotheses along the batch dim, the result owns its tensors. """
        if self.k is not None:
            self.k = self.k.index_select(0, parent_idx)
            self.v = self.v.index_select(0, parent_idx)


class CrossKVSlot:
    """ One decoder layer's projected cross-attention K / V, filled by that layer inside its forward (once). """
    __slots__ = ('k', 'v')

    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None


class CrossMemory:
    """ Per image cross-attention memory: prepared encoder states, mask, and per-layer projected K / V slots.

    The batch dim is the image batch. It is never reordered or expanded for beams / multiple prompts per image,
    cross-attention folds the hypotheses of an image into the query length instead. That relies on the hypothesis
    batch being image-major: `num_hypotheses = num_images * group_size`, w/ the hypotheses of image i at rows
    [i * group_size, (i + 1) * group_size). Selecting / regrouping images needs a new memory.
    """
    __slots__ = ('states', 'mask', 'slots', 'batch_size')

    def __init__(self, states: torch.Tensor, mask, num_layers: int, batch_size: int):
        self.states: Optional[torch.Tensor] = states  # prepared encoder output (packed for varlen), freed once projected
        self.mask = mask  # CrossAttnMask or None (no padding)
        self.slots: List[CrossKVSlot] = [CrossKVSlot() for _ in range(num_layers)]
        self.batch_size = batch_size  # number of images


class PrefixMemory:
    """ Per image memory of prefix-LM / decoder-only models: image prefix embeddings, validity, and (optional)
    model specific position info. The prefix K / V live in the sequence state (self-attention cache). """
    __slots__ = ('embeds', 'mask', 'pos', 'text_start')

    def __init__(self, embeds: torch.Tensor, mask: torch.Tensor, pos=None, text_start=None):
        self.embeds = embeds  # [B, P, C]
        self.mask = mask  # [B, P] bool
        self.pos = pos  # model specific prefix position ids
        self.text_start = text_start  # model specific first text position per image


class DecodeState:
    """ Per hypothesis decoding state, updated in place by `model.step()`.

    Attributes:
        cache: per-layer self-attention K / V caches
        attn_mask: [N, S] bool key validity of the cached sequence (image prefix, left padded prompt, generated)
        pos: [N] position id of the last token
        memory: per image memory (shared by reference, see CrossMemory for the batch layout invariant)
        prefix_len: length of the image prefix held in the self-attention cache (prefix-LM models)
    """
    __slots__ = ('cache', 'attn_mask', 'pos', 'memory', 'prefix_len')

    def __init__(self, cache: List[LayerKVCache], attn_mask: torch.Tensor, pos: torch.Tensor, memory=None,
                 prefix_len: int = 0):
        self.cache = cache
        self.attn_mask = attn_mask
        self.pos = pos
        self.memory = memory
        self.prefix_len = prefix_len

    @property
    def batch_size(self) -> int:
        return self.attn_mask.shape[0]

    def reorder(self, parent_idx: torch.Tensor) -> 'DecodeState':
        """ Select parent hypotheses for the next step, in place.

        `parent_idx` [N'] indexes the current hypotheses, indices may repeat (several children of one parent) and
        N' may differ from N (expansion to beams). Children get independent sequence state. The image memory is
        untouched, so every child must stay in its parent's image group (see CrossMemory).
        """
        for c in self.cache:
            c.reorder(parent_idx)
        self.attn_mask = self.attn_mask.index_select(0, parent_idx)
        self.pos = self.pos.index_select(0, parent_idx)
        return self
