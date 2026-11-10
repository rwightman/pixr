""" Parameter -> (component, layer id) maps for per component LR scaling and layer-wise LR decay.

Components:
    'encoder': image encoder (timm trunk, layer ids from its group_matcher)
    'bridge': image <-> text connection (decoder cross-attention, prefix projector)
    'decoder': text decoder / LM trunk

Layer ids increase from input (0) to output (num_layers - 1) within each component.
"""
import re
from typing import Dict, List, Tuple

from torch import nn

LayerMap = Dict[str, Tuple[str, int, int]]  # param name -> (component, layer id, num layers in component)

BRIDGE_PATTERNS = ('encoder_attn', 'cross_attn')  # decoder layer cross-attention (+ its norms)
_DECODER_BOTTOM = ('embed_positions', 'layernorm_embedding', 'embed_norm')
_LAYER_RE = re.compile(r'(?:^|\.)(?:layers|blocks)\.(\d+)\.')


def timm_layer_map(module: nn.Module, prefix: str, component: str = 'encoder') -> LayerMap:
    """ Layer ids for a timm model via its group_matcher (e.g. stem / embeds -> 0, blocks -> 1..N, head -> N + 1). """
    names = [n for n, _ in module.named_parameters()]
    if hasattr(module, 'group_matcher'):
        from timm.models import group_parameters
        layer_map = group_parameters(module, module.group_matcher(coarse=False), reverse=True)
        num_layers = max(layer_map.values()) + 1
    else:
        layer_map, num_layers = {}, 1
    return {prefix + n: (component, layer_map.get(n, num_layers - 1), num_layers) for n in names}


def decoder_layer_map(
        module: nn.Module,
        prefix: str,
        component: str = 'decoder',
        bridge: bool = True,
        bottom: Tuple[str, ...] = _DECODER_BOTTOM,
) -> LayerMap:
    """ Layer ids for a decoder / trunk w/ `layers.N.` blocks.

    Position embeddings and embedding norms are the bottom layer, blocks 1..N, everything else (final norms, head,
    and tied token embeddings, which act as the output projection and hold the new task tokens) the top layer.
    Cross-attention params in blocks are assigned to the 'bridge' component (w/ the block's layer id) if `bridge`.
    """
    num_blocks = len(module.layers)
    num_layers = num_blocks + 2
    out = {}
    for n, _ in module.named_parameters():
        m = _LAYER_RE.search(n)
        if m:
            layer_id = int(m.group(1)) + 1
        elif n.startswith(bottom):
            layer_id = 0
        else:
            layer_id = num_layers - 1
        comp = 'bridge' if bridge and m and any(p in n for p in BRIDGE_PATTERNS) else component
        out[prefix + n] = (comp, layer_id, num_layers)
    return out


def complete_layer_map(model: nn.Module, layer_map: LayerMap, component: str = 'bridge') -> LayerMap:
    """ Assign any params not in the map (e.g. encoder output projections, prefix projector) to `component`. """
    for n, _ in model.named_parameters():
        if n not in layer_map:
            layer_map[n] = (component, 0, 1)
    return layer_map


def block_modules(module: nn.Module, prefix: str, attr: str = 'blocks') -> List[Tuple[str, nn.Module]]:
    """ (name, block) pairs of a module's block list, the per block units for FSDP2 / activation checkpointing. """
    blocks = getattr(module, attr, None)
    if blocks is None:
        return []
    return [(f'{prefix}{attr}.{i}', blk) for i, blk in enumerate(blocks)]
