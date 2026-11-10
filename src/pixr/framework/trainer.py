""" Trainer: optimization loop for a TrainingTask (optimizer, schedule, AMP, accumulation, DDP / compile, EMA,
logging, and train state checkpointing). """
import logging
import re
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch

import timm.utils
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from .config import OptimizationCfg, TrainerCfg
from .device import DeviceEnv
from .monitor import Monitor
from .task import TrainingTask

_logger = logging.getLogger(__name__)


def param_groups_lr_scaled(model, opt_cfg: OptimizationCfg, verbose: bool = False):
    """ Param groups w/ per component LR scales, optional layer-wise LR decay, and no weight decay for norm, bias,
    and model specified (no_weight_decay()) params.

    Uses the model's param_layer_map() ({name: (component, layer id, num layers)}), params of models w/o one are
    assigned by name prefix ('image_encoder.' -> encoder, else decoder) w/o layer decay.
    """
    no_wd = set(model.no_weight_decay()) if hasattr(model, 'no_weight_decay') else set()
    if hasattr(model, 'param_layer_map'):
        layer_map = model.param_layer_map()
    else:
        layer_map = {
            n: ('encoder' if n.startswith('image_encoder.') else 'decoder', 0, 1) for n, _ in model.named_parameters()}
    component_scale = {
        'encoder': opt_cfg.encoder_lr_scale,
        'bridge': opt_cfg.bridge_lr_scale,
        'decoder': opt_cfg.decoder_lr_scale,
    }
    component_decay = {'encoder': opt_cfg.encoder_layer_decay, 'decoder': opt_cfg.decoder_layer_decay}
    overrides = [(re.compile(k), v) for k, v in (opt_cfg.lr_scale_overrides or {}).items()]

    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        component, layer_id, num_layers = layer_map[name]
        scale = component_scale[component]
        decay = component_decay.get(component)
        if decay is not None:
            scale *= decay ** (num_layers - 1 - layer_id)
        for regex, s in overrides:
            if regex.search(name):
                scale *= s
                break
        use_wd = not (param.ndim <= 1 or name.endswith('.bias') or name in no_wd)
        key = (component, round(scale, 12), use_wd)
        if key not in groups:
            groups[key] = {
                'params': [],
                'param_names': [],
                'weight_decay': opt_cfg.weight_decay if use_wd else 0.,
                'lr_scale': scale,
            }
        groups[key]['params'].append(param)
        groups[key]['param_names'].append(name)

    if verbose:
        summary = {}
        for (component, scale, _), g in groups.items():
            s = summary.setdefault(component, {'num_params': 0, 'scales': set()})
            s['num_params'] += sum(p.numel() for p in g['params'])
            s['scales'].add(scale)
        for component, s in summary.items():
            _logger.info(
                f'LR param groups [{component}]: {s["num_params"] / 1e6:.1f}M params, '
                f'lr_scale {min(s["scales"]):.3g} - {max(s["scales"]):.3g} ({len(s["scales"])} distinct)')
    for g in groups.values():
        del g['param_names']
    return list(groups.values())


class Trainer:
    def __init__(
            self,
            task: TrainingTask,
            cfg: TrainerCfg,
            device_env: DeviceEnv,
            monitor: Optional[Monitor] = None,
    ):
        self.task = task
        self.cfg = cfg
        self.device_env = device_env
        self.monitor = monitor
        self.step_idx = 0  # optimizer steps
        self.batch_idx = 0
        self.interval_idx = 0
        self.num_steps_per_interval = None
        self.mean_batch_size = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self._rate_time = None
        self._rate_samples = 0

    @property
    def model(self):
        """ The unwrapped model (w/o DDP / compile wrappers). """
        return self.task.get_trainable_module()

    @property
    def total_steps(self) -> int:
        return self.cfg.num_intervals * self.num_steps_per_interval

    def setup(
            self,
            num_batches_per_interval: int,
            resume: str = '',
            mean_batch_size: Optional[float] = None,
    ):
        self.mean_batch_size = mean_batch_size
        self._prepare_task()

        opt_cfg = self.cfg.opt
        opt_kwargs = {}
        if opt_cfg.betas is not None:
            opt_kwargs['betas'] = tuple(opt_cfg.betas)
        if opt_cfg.momentum is not None:
            opt_kwargs['momentum'] = opt_cfg.momentum
        params = param_groups_lr_scaled(self.model, opt_cfg, verbose=self.device_env.is_primary())
        self.optimizer = create_optimizer_v2(
            params,
            opt_cfg.optimizer,
            lr=opt_cfg.learning_rate,
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
            **opt_kwargs,
        )
        self.scaler = None
        if self.task.amp_dtype == torch.float16:
            assert not self.task.fsdp_enabled, 'FSDP w/ float16 is not supported, use bfloat16.'
            self.scaler = timm.utils.NativeScaler()

        self.num_steps_per_interval = num_batches_per_interval // opt_cfg.grad_accum_steps
        warmup_steps = int(self.cfg.num_warmup_intervals * self.num_steps_per_interval)
        # schedule in steps (updates_per_epoch=1) so fractional warmup intervals are supported
        self.scheduler, _ = create_scheduler_v2(
            self.optimizer,
            opt_cfg.scheduler,
            min_lr=opt_cfg.min_learning_rate,
            warmup_lr=opt_cfg.warmup_learning_rate,
            warmup_epochs=warmup_steps,
            num_epochs=self.total_steps,
            cooldown_epochs=0,
            step_on_epochs=False,
            updates_per_epoch=1,
        )

        if resume:
            from pixr.utils import load_checkpoint
            self.load_state_dict(load_checkpoint(resume))
            _logger.info(f'Resumed from {resume} at interval {self.interval_idx}, step {self.step_idx}.')
        self.scheduler.step_update(self.step_idx)

        if self.cfg.compile and self.cfg.compile_strategy == 'task':
            _logger.info('Compiling task training forward.')
            self.task.compile(target='task', **self._compile_kwargs())

    def _compile_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(backend=self.cfg.compile_backend, mode=self.cfg.compile_mode, dynamic=self.cfg.compile_dynamic)
        return {k: v for k, v in kwargs.items() if v is not None}

    def _prepare_task(self):
        """ Grad checkpointing, EMA, compile, and DDP / FSDP2 wrapping, ordered as per OpenCLIP's train setup. """
        cfg = self.cfg
        task = self.task
        distributed = self.device_env.world_size > 1
        use_fsdp = cfg.fsdp and distributed
        if cfg.fsdp and not distributed:
            _logger.warning('FSDP requires distributed mode, ignoring trainer.fsdp for single process training.')
        assert cfg.compile_strategy in ('task', 'model', 'blocks'), f'Unknown compile strategy {cfg.compile_strategy}'
        task.to(self.device_env.device)

        if cfg.grad_checkpointing and not use_fsdp:
            # w/ FSDP, composable AC is applied per block inside prepare_fsdp() (after per block compile)
            self.model.set_grad_checkpointing()
        if use_fsdp and cfg.compile and cfg.grad_checkpointing and torch.get_num_threads() > 1:
            # DataLoader pin-memory threads call set_num_threads(1), which can also reach the autograd device thread.
            # The checkpoint recompute then runs under a different thread count than the forward, recompiling the
            # block and failing w/ a CheckpointError. torchrun avoids this by default by setting OMP_NUM_THREADS=1.
            _logger.warning(
                f'Reducing torch num_threads from {torch.get_num_threads()} to 1 to avoid a CheckpointError '
                'w/ FSDP, compile, and grad checkpointing.')
            torch.set_num_threads(1)

        if cfg.ema_decay:
            # before compile and distributed wrapping, so the copy is of the plain model
            task.setup_ema(cfg.ema_decay)

        compile_kwargs = self._compile_kwargs()
        if cfg.compile and cfg.compile_strategy in ('model', 'blocks'):
            if use_fsdp:
                _logger.info(f'Compile strategy {cfg.compile_strategy} w/ FSDP uses prepare_fsdp() per block compile.')
            else:
                _logger.info(f'Compiling ({cfg.compile_strategy}) before distributed wrapping.')
                task.compile(target=cfg.compile_strategy, **compile_kwargs)

        if use_fsdp:
            from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
            # autocast is suppressed w/ FSDP, the policy casts params / forward inputs, gradients reduced in fp32
            mp_policy = MixedPrecisionPolicy(param_dtype=task.amp_dtype, reduce_dtype=torch.float32)
            _logger.info(f'FSDP2: MixedPrecisionPolicy(param_dtype={task.amp_dtype}, reduce_dtype=torch.float32).')
            task.prepare_fsdp(
                reshard_after_forward=cfg.fsdp_reshard_after_forward,
                mp_policy=mp_policy,
                offload_policy=CPUOffloadPolicy() if cfg.fsdp_offload_cpu else None,
                compile_blocks=cfg.compile,
                compile_kwargs=compile_kwargs,
                grad_checkpointing=cfg.grad_checkpointing,
            )
        elif distributed:
            if cfg.compile:
                # The DDP dynamo optimizer splits the graph into submodules at gradient-bucket boundaries. That breaks
                # under grad checkpointing and dynamic cross-bucket sequence lengths (NaFlex patches + variable text
                # length), Inductor can't recover the symbol ("expected [sN] to have been codegen-ed"). Disable the
                # split so the whole forward compiles as one symbol-consistent graph.
                torch._dynamo.config.optimize_ddp = False
            task.prepare_distributed(device_ids=[self.device_env.device])

    def _loss_scale(self, batch) -> float:
        mode = self.cfg.opt.loss_batch_scale
        if mode == 'none' or not self.mean_batch_size:
            return 1.0
        ratio = self.task.batch_size(batch) / self.mean_batch_size
        if mode == 'linear':
            return ratio
        assert mode == 'sqrt', f'Unknown loss_batch_scale {mode}'
        return ratio ** 0.5

    def _get_rate(self) -> Optional[float]:
        now = time.time()
        rate = None
        if self._rate_time is not None:
            rate = self._rate_samples / max(now - self._rate_time, 1e-6)
        self._rate_time = now
        self._rate_samples = 0
        return rate

    def get_current_lr(self) -> float:
        return max(g['lr'] for g in self.optimizer.param_groups)

    def train_interval(self, loader) -> Dict[str, float]:
        opt_cfg = self.cfg.opt
        accum = opt_cfg.grad_accum_steps
        ddp_module = self.task.trainable_module
        has_no_sync = isinstance(ddp_module, torch.nn.parallel.DistributedDataParallel)
        self.task.train()
        self.optimizer.zero_grad(set_to_none=True)
        start = time.time()
        num_samples = 0
        for interval_batch_idx, batch in enumerate(loader):
            batch = self.task.prepare_batch(batch)
            batch_size = self.task.batch_size(batch) * self.device_env.world_size
            num_samples += batch_size
            self._rate_samples += batch_size
            need_update = (interval_batch_idx + 1) % accum == 0
            # disable gradient sync for all but the last accumulation step, FSDP2: set_requires_gradient_sync,
            # DDP: no_sync context manager
            if not need_update:
                self.task.set_requires_gradient_sync(False)
            sync_ctx = ddp_module.no_sync if (has_no_sync and not need_update) else nullcontext
            with sync_ctx():
                with self.task.autocast():
                    outputs, losses = self.task(batch)
                loss = losses['loss'] * self._loss_scale(batch)
                if accum > 1:
                    loss = loss / accum
                self._backward(loss, need_update)
            if not need_update:
                self.task.set_requires_gradient_sync(True)
            self.batch_idx += 1
            if not need_update:
                continue
            self.step_idx += 1
            self.scheduler.step_update(self.step_idx)
            self.optimizer.zero_grad(set_to_none=True)
            self.task.update_ema()
            self._after_step(batch, outputs, losses)
        self.interval_idx += 1
        if self.monitor is not None:
            self.monitor.log_phase('train', self.interval_idx - 1)
        elapsed = time.time() - start
        return {'samples_per_sec': num_samples / max(elapsed, 1e-6)}

    def _backward(self, loss, need_update: bool):
        params = self.model.parameters()
        if self.scaler is not None:
            self.scaler(
                loss,
                self.optimizer,
                clip_grad=self.cfg.opt.clip_grad_value,
                clip_mode=self.cfg.opt.clip_grad_mode,
                parameters=params,
                need_update=need_update,
            )
            return
        loss.backward()
        if need_update:
            if self.cfg.opt.clip_grad_value is not None:
                # clip_grad_norm_ handles FSDP2 DTensor grads (the total norm is reduced across shards)
                timm.utils.dispatch_clip_grad(params, value=self.cfg.opt.clip_grad_value, mode=self.cfg.opt.clip_grad_mode)
            self.optimizer.step()

    def _after_step(self, batch, outputs, losses):
        metrics = None
        eval_data = None
        if self.cfg.metrics_frequency and self.step_idx % self.cfg.metrics_frequency == 0:
            # w/ FSDP all ranks gather the full params (collective), then the primary runs the metrics alone
            with self.task.full_params():
                if self.device_env.is_primary():
                    result = self.task.train_metrics(batch, outputs)
                    if result is not None:
                        metrics, eval_data = result
        if self.monitor is not None and (metrics is not None or self.step_idx % self.cfg.log_frequency == 0):
            extra = {k: v.item() for k, v in losses.items() if k != 'loss' and torch.is_tensor(v)}
            self.monitor.log_step(
                'train',
                step_idx=self.step_idx,
                step_end_idx=self.total_steps,
                interval=self.interval_idx,
                loss=losses['loss'].item(),
                lr=self.get_current_lr(),
                rate=self._get_rate(),
                metrics={**extra, **(metrics or {})} or None,
                eval_data=eval_data,
            )

    def _optimizer_state_dict(self) -> Dict[str, Any]:
        if not self.task.fsdp_enabled:
            return self.optimizer.state_dict()
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, StateDictOptions
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        return get_optimizer_state_dict(self.model, self.optimizer, options=options)

    def _load_optimizer_state_dict(self, state: Dict[str, Any]):
        # FSDP optimizer state is keyed by param name, DDP / single device by param index, not interchangeable
        fqn_keyed = any(isinstance(k, str) for k in state.get('state', {}))
        if fqn_keyed != self.task.fsdp_enabled:
            _logger.warning('Optimizer state saved w/ a different FSDP / DDP setting, not restored.')
            return
        if not self.task.fsdp_enabled:
            self.optimizer.load_state_dict(state)
            return
        from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict, StateDictOptions
        set_optimizer_state_dict(
            self.model, self.optimizer, optim_state_dict=state, options=StateDictOptions(full_state_dict=True))

    def state_dict(self) -> Dict[str, Any]:
        """ Train state, w/ FSDP this is a collective (call on all ranks) and is only populated on rank 0. """
        state = self.task.model_state_dict()
        state['optimizer'] = self._optimizer_state_dict()
        state['scheduler'] = self.scheduler.state_dict()
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        state.update(step_idx=self.step_idx, batch_idx=self.batch_idx, interval_idx=self.interval_idx)
        state.update(self.task.extra_state())
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        self.task.load_model_state_dict(state)
        if 'optimizer' in state:
            self._load_optimizer_state_dict(state['optimizer'])
        # NOTE the LR schedule is not restored from the checkpoint, it is fully determined by the current config +
        # step_idx (restoring timm scheduler state would revert the schedule length if num_intervals changed)
        if 'scaler' in state and self.scaler is not None:
            self.scaler.load_state_dict(state['scaler'])
        self.step_idx = state.get('step_idx', self.step_idx)
        self.batch_idx = state.get('batch_idx', self.batch_idx)
        self.interval_idx = state.get('interval_idx', self.interval_idx)
