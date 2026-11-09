import abc
import logging
import math
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch

import timm.utils
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from pixparse.utils import clean_state_dict
from .config import TaskTrainCfg, TaskEvalCfg, TaskCfg
from .device import DeviceEnv
from .monitor import Monitor


_logger = logging.getLogger(__name__)


def _amp_dtype(cfg: TaskCfg):
    if not cfg.amp:
        return None
    if cfg.dtype in ('float16', 'fp16', 'half'):
        return torch.float16
    return torch.bfloat16


class Task(abc.ABC):
    def __init__(
            self,
            device_env: DeviceEnv,
            monitor: Monitor = None,
    ):
        self.device_env = device_env
        self.monitor = monitor

    @abc.abstractmethod
    def step(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def end(self):
        pass


class TaskEval(Task):
    def __init__(
            self,
            cfg: TaskEvalCfg,
            device_env: DeviceEnv,
            monitor: Monitor = None,
    ):
        super().__init__(device_env=device_env, monitor=monitor)
        self.cfg = cfg
        self.amp_dtype = _amp_dtype(cfg)

    def autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device_env.device.type, dtype=self.amp_dtype)

    def setup(self, *args, **kwargs):
        pass


def param_groups_encoder_decoder(model, weight_decay: float, encoder_lr_scale: float = 1.0):
    """ Param groups w/ no weight decay for norm, bias, embedding params, and optional encoder LR scaling. """
    no_wd = set(model.no_weight_decay()) if hasattr(model, 'no_weight_decay') else set()
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_encoder = name.startswith('image_encoder.')
        decay = not (param.ndim <= 1 or name.endswith('.bias') or name in no_wd)
        key = (is_encoder, decay)
        if key not in groups:
            groups[key] = {
                'params': [],
                'weight_decay': weight_decay if decay else 0.,
                'lr_scale': encoder_lr_scale if is_encoder else 1.0,
            }
        groups[key]['params'].append(param)
    return list(groups.values())


class TaskTrain(Task):
    def __init__(
            self,
            cfg: TaskTrainCfg,
            device_env: DeviceEnv,
            monitor: Monitor = None,
    ):
        super().__init__(
            device_env=device_env,
            monitor=monitor,
        )
        self.cfg = cfg
        self.num_intervals = cfg.num_intervals
        self.num_warmup_intervals = cfg.num_warmup_intervals
        self.log_frequency = cfg.log_frequency
        self.metrics_frequency = cfg.metrics_frequency
        self.num_steps_per_interval = None  # uninitialized, needs dataset info

        self.step_idx = 0  # current step (aka optimizer, scheduler update)
        self.batch_idx = 0  # current train batch (step % accum steps)
        self.interval_idx = 0  # interval (aka epoch or restorable period-between-checkpoints)
        self.interval_batch_idx = 0  # batch within current interval

        self.model = None
        self.amp_dtype = _amp_dtype(cfg)
        self.has_no_sync = False

        # optimization state initialized in setup()
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

    def autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device_env.device.type, dtype=self.amp_dtype)

    @property
    def unwrapped_model(self):
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        return model

    def setup(
            self,
            num_batches_per_interval: int,
            resume: Optional[str] = "",
    ):
        """
        Training duration is 'interval' based. An interval is a full dataset epoch, or a fixed # of samples
        for large (resampled / streamed) datasets. LR is updated per optimizer step, num_batches_per_interval
        is required from the data pipeline to translate intervals -> steps.

        Args:
            num_batches_per_interval: pass info from data pipeline for LR scheduling
            resume: path of checkpoint to resume training (model + optimizer + schedule state) from
        """
        self.model.to(self.device_env.device)
        if self.cfg.opt.grad_checkpointing:
            self.model.set_grad_checkpointing()

        self._setup_optimization(num_batches_per_interval=num_batches_per_interval)

        if resume:
            from pixparse.utils import load_checkpoint
            state_dict = load_checkpoint(resume)
            self.load_state_dict(state_dict, restore_optimizer_state=True, restore_scheduler_state=True)
            _logger.info(f'Resumed from {resume} at interval {self.interval_idx}, step {self.step_idx}.')

        if self.device_env.world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device_env.device],
            )
            self.has_no_sync = True

        if self.cfg.compile:
            self.model = torch.compile(self.model)

    def _setup_optimization(self, num_batches_per_interval):
        opt_cfg = self.cfg.opt
        opt_kwargs = {}
        if opt_cfg.betas is not None:
            opt_kwargs["betas"] = tuple(opt_cfg.betas)
        if opt_cfg.momentum is not None:
            opt_kwargs["momentum"] = opt_cfg.momentum
        if opt_cfg.layer_decay is not None:
            params = self.model
            opt_kwargs['layer_decay'] = opt_cfg.layer_decay
        else:
            params = param_groups_encoder_decoder(
                self.model,
                weight_decay=opt_cfg.weight_decay,
                encoder_lr_scale=opt_cfg.encoder_lr_scale,
            )
        self.optimizer = create_optimizer_v2(
            params,
            opt_cfg.optimizer,
            lr=opt_cfg.learning_rate,
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
            **opt_kwargs,
        )

        if self.amp_dtype == torch.float16:
            self.scaler = timm.utils.NativeScaler()
        else:
            self.scaler = None

        self.num_steps_per_interval = num_batches_per_interval // opt_cfg.grad_accum_steps
        total_steps = self.num_intervals * self.num_steps_per_interval
        warmup_steps = int(self.num_warmup_intervals * self.num_steps_per_interval)
        # schedule in steps (updates_per_epoch=1) so fractional warmup intervals are supported
        self.scheduler, _ = create_scheduler_v2(
            self.optimizer,
            opt_cfg.scheduler,
            min_lr=opt_cfg.min_learning_rate,
            warmup_lr=opt_cfg.warmup_learning_rate,
            warmup_epochs=warmup_steps,
            num_epochs=total_steps,
            cooldown_epochs=0,
            step_on_epochs=False,
            updates_per_epoch=1,
        )
        self.scheduler.step_update(self.step_idx)

    @property
    def total_steps(self):
        return self.num_intervals * self.num_steps_per_interval

    def interval_start(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.interval_batch_idx = 0

    def interval_end(self):
        self.monitor.log_phase('train', self.interval_idx)
        self.interval_idx += 1

    def step(self, sample: Dict[str, Any]) -> Tuple[Union[torch.Tensor, Dict[str, Any]], ...]:
        output, loss, updated = self._update(sample)
        self._after_update(updated)
        return output, loss

    def _update(self, sample: Dict[str, Any]):
        accum_steps = self.cfg.opt.grad_accum_steps
        need_update = (self.interval_batch_idx + 1) % accum_steps == 0
        sync_ctx = self.model.no_sync if (self.has_no_sync and not need_update) else nullcontext
        with sync_ctx():
            output, loss = self._forward(sample)
            train_loss = self._loss(loss)
            if accum_steps > 1:
                train_loss = train_loss / accum_steps
            self._backward(train_loss, need_update=need_update)
        return output, loss, need_update

    def _after_update(self, updated=True):
        self.batch_idx += 1
        self.interval_batch_idx += 1
        if not updated:
            return
        self.step_idx += 1
        self.scheduler.step_update(self.step_idx)
        self.optimizer.zero_grad(set_to_none=True)

    @abc.abstractmethod
    def _forward(self, sample: Dict[str, Any]):
        """ model + loss forward pass
        """

    def _backward(self, loss, need_update=True):
        """ backward pass w/ gradient scaling & clipping
        """
        if self.scaler is not None:
            self.scaler(
                loss,
                self.optimizer,
                clip_grad=self.cfg.opt.clip_grad_value,
                clip_mode=self.cfg.opt.clip_grad_mode,
                parameters=self.model.parameters(),
                need_update=need_update,
            )
        else:
            loss.backward()
            if need_update:
                if self.cfg.opt.clip_grad_value is not None:
                    timm.utils.dispatch_clip_grad(
                        self.model.parameters(),
                        value=self.cfg.opt.clip_grad_value,
                        mode=self.cfg.opt.clip_grad_mode,
                    )
                self.optimizer.step()

    def _loss(self, loss):
        # extract the combined train loss for backward
        if isinstance(loss, (tuple, list)):
            return loss[0]
        elif isinstance(loss, dict):
            return loss['loss']
        return loss

    def after_step(self, sample, output, loss):
        """ operations to perform after step
        Anything that
          * needs to move tensor -> cpu (ie .item())
          * needs to execute code that is not torch.compile() safe
          * perform logging / io, etc
        """
        if self.step_idx % self.log_frequency == 0 and self.interval_batch_idx % self.cfg.opt.grad_accum_steps == 0:
            self.monitor.log_step(
                'train',
                step_idx=self.step_idx,
                step_end_idx=self.total_steps,
                interval=self.interval_idx,
                loss=self._loss(loss).item(),
                lr=self.get_current_lr(),
            )

    def get_current_lr(self):
        lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
        return max(lrl)

    def extra_state(self) -> Dict[str, Any]:
        """ Extra (non-tensor) state to store in checkpoints, e.g. model config, tokenizer tokens. """
        return {}

    def state_dict(self):
        state_dicts = {}
        state_dicts['model'] = self.unwrapped_model.state_dict()
        state_dicts['optimizer'] = self.optimizer.state_dict()
        if hasattr(self.scheduler, 'state_dict'):
            state_dicts['scheduler'] = self.scheduler.state_dict()
        if self.scaler is not None:
            state_dicts['scaler'] = self.scaler.state_dict()
        state_dicts['step_idx'] = self.step_idx
        state_dicts['batch_idx'] = self.batch_idx
        state_dicts['interval_idx'] = self.interval_idx
        state_dicts.update(self.extra_state())
        return state_dicts

    def load_state_dict(
            self,
            state_dict,
            start_interval=None,
            restore_optimizer_state=True,
            restore_scheduler_state=True,
    ):
        # assume state_dict contains only model if key not present
        state_dict_model = clean_state_dict(state_dict.get('model', state_dict))
        self.unwrapped_model.load_state_dict(state_dict_model)

        if 'optimizer' not in state_dict:
            return

        # restore optimization related state
        if restore_optimizer_state:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            if 'scaler' in state_dict and self.scaler is not None:
                self.scaler.load_state_dict(state_dict['scaler'])
        if restore_scheduler_state:
            # restore LR schedule / step / interval related state
            if 'scheduler' in state_dict:
                self.scheduler.load_state_dict(state_dict['scheduler'])
            self.step_idx = state_dict.get('step_idx', self.step_idx)
            self.batch_idx = state_dict.get('batch_idx', self.batch_idx)
            if start_interval is not None:
                self.interval_idx = start_interval
            else:
                self.interval_idx = state_dict.get('interval_idx', self.interval_idx)
            self.scheduler.step_update(self.step_idx)
