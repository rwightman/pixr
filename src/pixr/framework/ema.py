""" Exponential moving average of model weights, for plain and FSDP2 sharded models.

Ported from OpenCLIP's FSDPModelEma (open_clip/task/fsdp_ema.py).
"""
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


def _clean_name(name: str) -> str:
    return name.replace('._orig_mod.', '.').removeprefix('_orig_mod.')


def _to_local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, 'to_local') else tensor


class ModelEma(nn.Module):
    """ EMA of a model held in a separate copy of it.

    Params are paired by name (aliased / tied params are updated once, torch.compile wrapped blocks in the trained
    model are matched to the plain EMA blocks). When the trained model is FSDP2 sharded, the copy must be sharded the
    same way (`TrainingTask.prepare_fsdp`): each rank then updates its local shards w/o communication, and evaluating
    `module` is a collective like any FSDP forward.
    """

    def __init__(self, module: nn.Module, decay: float = 0.9999):
        super().__init__()
        self.module = module
        self.decay = decay
        self.module.eval()
        self.module.requires_grad_(False)
        # Non-persistent buffers are derived, not learned (lerping the -inf entries of an attn mask would give NaN).
        persistent = {_clean_name(k) for k in self.module.state_dict()}
        self._buffer_names = [n for n, _ in self._named_tensors(self.module.named_buffers()) if n in persistent]

    @staticmethod
    def _named_tensors(named: Iterable[Tuple[str, torch.Tensor]]) -> List[Tuple[str, torch.Tensor]]:
        return [(_clean_name(n), t) for n, t in named]

    @torch.no_grad()
    def init_fsdp(self):
        """ Run FSDP's lazy init and allocate its unsharded storage now (call after sharding `module`).

        The first forward would otherwise happen inside eval, w/ inference_mode, FSDP would create the sharded params
        and unsharded storage as inference tensors and the next `update()` would fail writing to them.
        """
        from torch.distributed.fsdp import FSDPModule
        if not isinstance(self.module, FSDPModule):
            return
        self.module._get_fsdp_state()._lazy_init()
        for mod in self.module.modules():
            if isinstance(mod, FSDPModule):
                mod.unshard()
                mod.reshard()

    def _param_pairs(self, model: nn.Module) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        model_params = dict(self._named_tensors(model.named_parameters()))
        ema_params = self._named_tensors(self.module.named_parameters())
        return [_to_local(p) for _, p in ema_params], [_to_local(model_params[n]) for n, _ in ema_params]

    def _buffer_pairs(self, model: nn.Module) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        model_buffers = dict(self._named_tensors(model.named_buffers()))
        ema_buffers = dict(self._named_tensors(self.module.named_buffers()))
        return [(ema_buffers[n], model_buffers[n]) for n in self._buffer_names]

    @torch.no_grad()
    def update(self, model: nn.Module, decay: Optional[float] = None):
        """ Update from `model` (unwrapped, must be sharded if FSDP, true after backward). """
        decay = self.decay if decay is None else decay
        ema_params, model_params = self._param_pairs(model)
        torch._foreach_lerp_(ema_params, model_params, weight=1. - decay)
        for ema_buffer, model_buffer in self._buffer_pairs(model):
            if ema_buffer.is_floating_point():
                ema_buffer.lerp_(model_buffer, 1. - decay)
            else:
                ema_buffer.copy_(model_buffer)

    @torch.no_grad()
    def set(self, model: nn.Module):
        """ Copy `model`'s weights into the EMA. """
        torch._foreach_copy_(*self._param_pairs(model))
        for ema_buffer, model_buffer in self._buffer_pairs(model):
            ema_buffer.copy_(model_buffer)
