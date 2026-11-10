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
        self.task_wrapped = task  # DDP / compile wrapped task used for training forward
        self._rate_time = None
        self._rate_samples = 0

    @property
    def model(self):
        return self.task.trainable_module

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
        self.task.to(self.device_env.device)
        if self.cfg.grad_checkpointing:
            self.model.set_grad_checkpointing()
        if self.cfg.ema_decay:
            self.task.setup_ema(self.cfg.ema_decay)

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
        self.scaler = timm.utils.NativeScaler() if self.task.amp_dtype == torch.float16 else None

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

        task = self.task
        if self.device_env.world_size > 1:
            task = torch.nn.parallel.DistributedDataParallel(task, device_ids=[self.device_env.device])
        if self.cfg.compile:
            task = torch.compile(task)
        self.task_wrapped = task

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
        has_no_sync = hasattr(self.task_wrapped, 'no_sync')
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
            sync_ctx = self.task_wrapped.no_sync if (has_no_sync and not need_update) else nullcontext
            with sync_ctx():
                with self.task.autocast():
                    outputs, losses = self.task_wrapped(batch)
                loss = losses['loss'] * self._loss_scale(batch)
                if accum > 1:
                    loss = loss / accum
                self._backward(loss, need_update)
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
                timm.utils.dispatch_clip_grad(params, value=self.cfg.opt.clip_grad_value, mode=self.cfg.opt.clip_grad_mode)
            self.optimizer.step()

    def _after_step(self, batch, outputs, losses):
        metrics = None
        eval_data = None
        if self.cfg.metrics_frequency and self.step_idx % self.cfg.metrics_frequency == 0 and \
                self.device_env.is_primary():
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

    def state_dict(self) -> Dict[str, Any]:
        state = self.task.model_state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        state['scheduler'] = self.scheduler.state_dict()
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        state.update(step_idx=self.step_idx, batch_idx=self.batch_idx, interval_idx=self.interval_idx)
        state.update(self.task.extra_state())
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        self.task.load_model_state_dict(state)
        if 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
        # NOTE the LR schedule is not restored from the checkpoint, it is fully determined by the current config +
        # step_idx (restoring timm scheduler state would revert the schedule length if num_intervals changed)
        if 'scaler' in state and self.scaler is not None:
            self.scaler.load_state_dict(state['scaler'])
        self.step_idx = state.get('step_idx', self.step_idx)
        self.batch_idx = state.get('batch_idx', self.batch_idx)
        self.interval_idx = state.get('interval_idx', self.interval_idx)
