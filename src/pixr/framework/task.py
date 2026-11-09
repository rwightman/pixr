""" Training task base.

A task wraps a model w/ its objective, batch contract, pre / post-processing, data pipeline configs, evaluation,
and task specific checkpoint state. It is an nn.Module whose forward runs the training forward (incl. loss) so the
whole step can be wrapped by DDP / torch.compile. The optimization loop (optimizer, schedule, accumulation, logging,
checkpoint I/O) lives in the Trainer.
"""
import abc
import copy
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from pixr.utils import clean_state_dict
from .config import TaskCfg
from .device import DeviceEnv


def _amp_dtype(cfg: TaskCfg):
    if not cfg.amp:
        return None
    if cfg.dtype in ('float16', 'fp16', 'half'):
        return torch.float16
    return torch.bfloat16


def move_to_device(x, device, non_blocking: bool = True):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in x.items()}
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], torch.Tensor):
        return type(x)(move_to_device(v, device, non_blocking) for v in x)
    return x


class TrainingTask(nn.Module, abc.ABC):
    """ Model + objective. Subclasses implement training_forward() and optionally evaluate() / train_metrics(). """
    task_name: str = ''

    def __init__(self, cfg: TaskCfg, model: nn.Module, device_env: DeviceEnv):
        super().__init__()
        self.cfg = cfg
        self.trainable_module = model
        self.trainable_module_ema: Optional[nn.Module] = None
        self.device_env = device_env
        self.amp_dtype = _amp_dtype(cfg)

    @property
    def device(self) -> torch.device:
        return self.device_env.device

    def autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    # --- batch / forward contract ---
    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {k: move_to_device(v, self.device) for k, v in batch.items()}

    def batch_size(self, batch: Dict[str, Any]) -> int:
        x = batch['image_input']
        return (x['patches'] if isinstance(x, dict) else x).shape[0]

    @abc.abstractmethod
    def training_forward(self, batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """ Returns (outputs, losses), losses['loss'] is the combined training loss. """

    def forward(self, batch: Dict[str, Any]):
        return self.training_forward(batch)

    # --- optional hooks ---
    def train_metrics(self, batch: Dict[str, Any], outputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """ Expensive train metrics on the primary process every `metrics_frequency` steps. """
        return None

    def evaluate(self, loader, use_ema: bool = False) -> Dict[str, Any]:
        raise NotImplementedError(f'Task {self.task_name} does not support evaluation.')

    def train_data_cfg(self):
        raise NotImplementedError

    def eval_data_cfg(self):
        raise NotImplementedError

    def extra_state(self) -> Dict[str, Any]:
        """ Extra (non-tensor) checkpoint state, e.g. model config, tokenizer tokens. """
        return {}

    # --- model access / EMA ---
    def get_trainable_module(self, use_ema: bool = False) -> nn.Module:
        if use_ema and self.trainable_module_ema is not None:
            return self.trainable_module_ema
        return self.trainable_module

    def setup_ema(self, decay: float):
        """ EMA copy of the model, kept out of the module tree (not wrapped by DDP / compile, not in parameters()). """
        self.ema_decay = decay
        ema = copy.deepcopy(self.trainable_module).eval().requires_grad_(False)
        object.__setattr__(self, 'trainable_module_ema', ema)

    @torch.no_grad()
    def update_ema(self):
        if self.trainable_module_ema is None:
            return
        for ema_v, v in zip(self.trainable_module_ema.state_dict().values(), self.trainable_module.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.lerp_(v.detach(), 1. - self.ema_decay)
            else:
                ema_v.copy_(v)

    def model_state_dict(self) -> Dict[str, Any]:
        state = {'model': self.trainable_module.state_dict()}
        if self.trainable_module_ema is not None:
            state['model_ema'] = self.trainable_module_ema.state_dict()
        return state

    def load_model_state_dict(self, state: Dict[str, Any]):
        self.trainable_module.load_state_dict(clean_state_dict(state.get('model', state)))
        if self.trainable_module_ema is not None and 'model_ema' in state:
            self.trainable_module_ema.load_state_dict(clean_state_dict(state['model_ema']))
