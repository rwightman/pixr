""" Training task base.

A task wraps a model w/ its objective, batch contract, pre / post-processing, data pipeline configs, evaluation,
and task specific checkpoint state. It is an nn.Module whose forward runs the training forward (incl. loss). The
optimization loop (optimizer, schedule, accumulation, logging, checkpoint I/O) lives in the Trainer.

Distributed / compile handling follows the OpenCLIP TrainingTask: the task owns DDP (`prepare_distributed`) or FSDP2
(`prepare_fsdp`) wrapping of `trainable_module` and torch.compile (`compile`) of the model, its blocks, or the task
forward. Task code calls `self.trainable_module(...)` for the (wrapped) training forward and
`self.get_trainable_module()` for the unwrapped model (attribute / method access, generation).
"""
import abc
import logging
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

from pixr.utils import clean_state_dict
from .config import TaskCfg
from .device import DeviceEnv
from .ema import ModelEma

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy

_logger = logging.getLogger(__name__)

# model methods run outside forward() that must trigger FSDP all-gather / reshard like a forward
_FSDP_FORWARD_METHODS = ('prefill', 'step')


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


def unwrap_model(model: nn.Module) -> nn.Module:
    """ Strip DDP and torch.compile wrappers (FSDP2 shards in place, so needs no unwrapping). """
    while True:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        elif isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
            model = model._orig_mod
        else:
            return model


def strip_compiled_keys(sd: Dict[str, Any]) -> Dict[str, Any]:
    """ Remove the `_orig_mod.` segments torch.compile wrappers (model or per block) insert into state dict keys. """
    return {k.replace('._orig_mod.', '.').removeprefix('_orig_mod.'): v for k, v in sd.items()}


class TrainingTask(nn.Module, abc.ABC):
    """ Model + objective. Subclasses implement training_forward() and optionally evaluate() / train_metrics(). """
    task_name: str = ''

    def __init__(self, cfg: TaskCfg, model: nn.Module, device_env: DeviceEnv):
        super().__init__()
        self.cfg = cfg
        self.trainable_module = model
        self.trainable_module_ema: Optional[ModelEma] = None
        self.device_env = device_env
        self.amp_dtype = _amp_dtype(cfg)
        self._fsdp_enabled = False
        self._fsdp_reshard_after_forward = True
        self._compiled_training_forward = None

    @property
    def device(self) -> torch.device:
        return self.device_env.device

    @property
    def fsdp_enabled(self) -> bool:
        return self._fsdp_enabled

    def autocast(self):
        # w/ FSDP2 the MixedPrecisionPolicy casts params / inputs, autocast is suppressed (as per OpenCLIP)
        if self.amp_dtype is None or self._fsdp_enabled:
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
        forward_fn = self._compiled_training_forward or self.training_forward
        return forward_fn(batch)

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
        """ The unwrapped model (or its EMA), use `trainable_module` directly for the wrapped training forward. """
        if use_ema and self.trainable_module_ema is not None:
            return self.trainable_module_ema.module
        return unwrap_model(self.trainable_module)

    def setup_ema(self, decay: float):
        """ EMA copy of the model, kept out of the module tree (not wrapped by DDP / compile, not in parameters()).

        Must be called before `prepare_fsdp()` (FSDP2 modules can't be deep-copied), which shards the copy too.
        """
        assert not self._fsdp_enabled, 'EMA must be set up before prepare_fsdp().'
        import copy
        ema = ModelEma(copy.deepcopy(unwrap_model(self.trainable_module)), decay=decay)
        object.__setattr__(self, 'trainable_module_ema', ema)

    @torch.no_grad()
    def update_ema(self):
        if self.trainable_module_ema is not None:
            self.trainable_module_ema.update(self.get_trainable_module())

    @property
    def has_ema(self) -> bool:
        return self.trainable_module_ema is not None

    # --- compile / distributed ---
    def compile(self, target: str = 'task', **compile_kwargs) -> 'TrainingTask':
        """ torch.compile w/o replacing the task object.

        Targets:
            'model': compile trainable_module (before DDP wrapping)
            'blocks': compile the model's blocks in place (before DDP wrapping), surrounding code (incl. inline grad
                checkpointing) stays eager, so checkpoint recompute is sequenced block by block by eager autograd
                instead of traced into the compiled backward (bounds compiled backward memory)
            'task': compile the task training forward (model + loss), after distributed wrapping
        """
        if target == 'model':
            self.trainable_module = torch.compile(self.trainable_module, **compile_kwargs)
        elif target == 'blocks':
            blocks = self._get_fsdp_shard_modules()
            if not blocks:
                raise ValueError("compile target 'blocks' found no blocks (model needs fsdp_shard_modules())")
            model = unwrap_model(self.trainable_module)
            for name, mod in blocks:
                parent_name, _, attr = name.rpartition('.')
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, attr, torch.compile(mod, **compile_kwargs))
        elif target == 'task':
            self._compiled_training_forward = torch.compile(self.training_forward, **compile_kwargs)
        else:
            raise ValueError(f'Unsupported task compile target: {target}')
        return self

    def prepare_distributed(self, device_ids: Optional[List] = None, **ddp_kwargs) -> 'TrainingTask':
        """ Wrap trainable_module w/ DistributedDataParallel. """
        assert not self._fsdp_enabled, 'Cannot wrap an FSDP2 sharded module w/ DDP, use FSDP or DDP, not both.'
        self.trainable_module = torch.nn.parallel.DistributedDataParallel(
            self.trainable_module, device_ids=device_ids, **ddp_kwargs)
        return self

    def _get_fsdp_shard_modules(self) -> List[Tuple[str, nn.Module]]:
        """ (name, module) blocks to wrap individually for FSDP2, composable checkpointing and 'blocks' compile. """
        model = unwrap_model(self.trainable_module)
        if hasattr(model, 'fsdp_shard_modules'):
            return model.fsdp_shard_modules()
        return []

    def prepare_fsdp(
            self,
            mesh: Optional['DeviceMesh'] = None,
            reshard_after_forward: bool = True,
            mp_policy: Optional['MixedPrecisionPolicy'] = None,
            offload_policy: Optional['CPUOffloadPolicy'] = None,
            compile_blocks: bool = False,
            compile_kwargs: Optional[Dict[str, Any]] = None,
            grad_checkpointing: bool = False,
    ) -> 'TrainingTask':
        """ Apply FSDP2 sharding to the trainable module: each shard block, then the root.

        Activation checkpointing is applied here (composable, per shard block) instead of the models' inline
        checkpointing. An EMA from setup_ema() is sharded the same way.
        """
        assert not isinstance(self.trainable_module, torch.nn.parallel.DistributedDataParallel), \
            'Cannot apply FSDP2 to a DDP wrapped module, use FSDP or DDP, not both.'
        model = unwrap_model(self.trainable_module)
        scalars = [n for n, p in model.named_parameters() if p.ndim == 0]
        assert not scalars, f'FSDP2 needs params w/ a dim to shard, found 0-D params {scalars}.'

        fsdp_kwargs = dict(mesh=mesh, reshard_after_forward=reshard_after_forward)
        if mp_policy is not None:
            fsdp_kwargs['mp_policy'] = mp_policy
        if offload_policy is not None:
            fsdp_kwargs['offload_policy'] = offload_policy

        shard_modules = self._get_fsdp_shard_modules()
        _logger.info(f'FSDP2: sharding {len(shard_modules)} blocks + root')
        if not shard_modules:
            _logger.warning('FSDP2: model defines no fsdp_shard_modules(), only the root will be sharded.')

        if grad_checkpointing:
            model.set_grad_checkpointing(False)  # composable AC per block replaces the inline checkpointing
            from torch.distributed._composable import checkpoint as composable_checkpoint

        # Required ordering: compile -> composable AC -> FSDP.
        #
        # Composable AC hooks must be registered on the OptimizedModule wrapper, NOT on the inner block. If hooks are
        # on the inner block, AC recomputation replays inside the compiled graph where FSDP DTensor metadata has
        # changed, causing "Recomputed values have different metadata" errors. With hooks on the OptimizedModule, AC
        # recomputation triggers at the module boundary (outside the compiled graph), which also triggers FSDP
        # all-gather hooks, so parameters are correctly gathered during recompute.
        wrapped_modules = []
        for name, mod in shard_modules:
            if compile_blocks:
                mod = torch.compile(mod, **(compile_kwargs or {}))
                parent_name, _, attr = name.rpartition('.')
                parent = model.get_submodule(parent_name) if parent_name else model
                parent.register_module(attr, mod)  # re-register so FSDP sees the compiled wrapper
            if grad_checkpointing:
                composable_checkpoint(mod)
            wrapped_modules.append((name, mod))
        if compile_blocks:
            _logger.info(f'FSDP2: compiled {len(wrapped_modules)} blocks w/ torch.compile')

        self._fully_shard_model(model, wrapped_modules, fsdp_kwargs)

        if self.trainable_module_ema is not None:
            # The EMA only runs eval forwards, so it always reshards and isn't compiled or checkpointed.
            ema_module = self.trainable_module_ema.module
            self._fully_shard_model(
                ema_module,
                [(name, ema_module.get_submodule(name)) for name, _ in shard_modules],
                {**fsdp_kwargs, 'reshard_after_forward': True},
            )
            self.trainable_module_ema.init_fsdp()

        self._fsdp_enabled = True
        self._fsdp_reshard_after_forward = reshard_after_forward
        return self

    @staticmethod
    def _fully_shard_model(root: nn.Module, shard_modules: List[Tuple[str, nn.Module]], fsdp_kwargs: dict):
        from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method
        for _, mod in shard_modules:
            fully_shard(mod, **fsdp_kwargs)
        # Root: remaining params (embeddings, norms, projections, heads). Left at FSDP's default (no reshard after
        # forward for the root), the fused LM loss uses the root's (tied) output projection after the model forward.
        root_kwargs = {k: v for k, v in fsdp_kwargs.items() if k != 'reshard_after_forward'}
        fully_shard(root, **root_kwargs)
        # generation entry points must all-gather / reshard like __call__
        for method_name in _FSDP_FORWARD_METHODS:
            if hasattr(root, method_name):
                register_fsdp_forward_method(root, method_name)

    def set_requires_gradient_sync(self, requires_sync: bool):
        """ FSDP2 equivalent of DDP no_sync() for gradient accumulation. """
        if self._fsdp_enabled:
            self.get_trainable_module().set_requires_gradient_sync(requires_sync)

    @contextmanager
    def full_params(self, use_ema: bool = False):
        """ Hold the full (unsharded) params of the model (or EMA) for the duration (FSDP2, collective on entry).

        Within, forwards issue no collectives, so ranks can run eval / generation independently (different numbers
        of batches or decode steps, or primary only). No-op w/o FSDP.
        """
        if not self._fsdp_enabled:
            yield
            return
        from torch.distributed.fsdp import FSDPModule
        root = self.get_trainable_module(use_ema)
        root._get_fsdp_state()._lazy_init()
        fsdp_modules = [m for m in root.modules() if isinstance(m, FSDPModule)]
        for m in fsdp_modules:
            m.set_reshard_after_forward(False, recurse=False)
            m.unshard()
        try:
            yield
        finally:
            restore = self._fsdp_reshard_after_forward if root is self.get_trainable_module() else True
            for m in fsdp_modules:
                m.reshard()
                if m is not root:  # the root keeps FSDP's default (no reshard after forward)
                    m.set_reshard_after_forward(restore, recurse=False)

    # --- checkpoint state ---
    def _module_state_dict(self, module: nn.Module) -> Dict[str, Any]:
        """ Full state dict for a module, gathering shards under FSDP (collective, populated on rank 0 only). """
        if not self._fsdp_enabled:
            return strip_compiled_keys(module.state_dict())
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        return strip_compiled_keys(get_model_state_dict(module, options=options))

    def _load_module_state_dict(self, module: nn.Module, state_dict: Dict[str, Any]):
        sd = strip_compiled_keys(clean_state_dict(state_dict))
        if self._fsdp_enabled:
            # DCP state dict FQNs are already free of torch.compile `_orig_mod` segments, load the clean names
            from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
            set_model_state_dict(module, sd, options=StateDictOptions(full_state_dict=True))
            return
        model_keys = list(module.state_dict().keys())
        if any('_orig_mod' in k for k in model_keys):
            # remap clean names onto the live model's keys when its blocks are compiled in place
            clean_to_model = {k.replace('._orig_mod.', '.').removeprefix('_orig_mod.'): k for k in model_keys}
            sd = {clean_to_model.get(k, k): v for k, v in sd.items()}
        module.load_state_dict(sd)

    def model_state_dict(self) -> Dict[str, Any]:
        state = {'model': self._module_state_dict(self.get_trainable_module())}
        if self.trainable_module_ema is not None:
            state['model_ema'] = self._module_state_dict(self.trainable_module_ema.module)
        return state

    def load_model_state_dict(self, state: Dict[str, Any]):
        self._load_module_state_dict(self.get_trainable_module(), state.get('model', state))
        if self.trainable_module_ema is not None:
            if 'model_ema' in state:
                self._load_module_state_dict(self.trainable_module_ema.module, state['model_ema'])
            else:
                _logger.warning('Checkpoint has no EMA weights, initializing the EMA from the loaded model weights.')
                self.trainable_module_ema.set(self.get_trainable_module())
