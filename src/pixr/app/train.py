""" pixr training app

Example (read pretraining on IDL streamed from the HF hub, 2 GPUs):
    python -m torch.distributed.run --nproc_per_node 2 -m pixr.app.train \\
        --task read --task.model.name cruller_naflex_siglip2_base \\
        --train-data.source "pipe:curl -s -f -L https://huggingface.co/datasets/pixparse/idl-wds/resolve/main/idl-train-0{0000..2999}.tar" \\
        --train-data.num-samples 1000000 --train-data.batch-size 16 --train-data.num-workers 16 \\
        --trainer.opt.learning-rate 2e-4
"""
import logging
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Optional

import torch

from chugr.common import DataArg, Serializable

from pixr.data import create_loader
from pixr.framework import DeviceEnv, Monitor, Trainer, TrainerCfg, setup_logging, random_seed
from pixr.task import get_task, get_task_cfgs
from pixr.utils import clean_name, get_latest_checkpoint
from pixr.utils.cli import parse_cfg

_logger = logging.getLogger('train')


@dataclass
class TrainCfg(Serializable):
    train_data: DataArg = field(default_factory=lambda: DataArg(source=''))
    eval_data: Optional[DataArg] = None
    task_name: str = 'read'  # selected via --task <name> (or task_name in a --config file)
    task: Any = None  # task config, type set by the selected task
    trainer: TrainerCfg = field(default_factory=TrainerCfg)

    experiment: Optional[str] = None  # experiment name, auto-generated if None
    output_dir: str = './output'
    log_filename: str = 'out.log'
    resume: str = ''  # resume training from checkpoint w/ full train state ('latest' for most recent)
    checkpoint_path: str = ''  # initialize model weights (and tokens / arch) from this checkpoint
    eval_frequency: int = 1  # evaluate every N intervals (if eval_data set)
    eval_ema: bool = False  # evaluate the EMA weights (if trainer.ema_decay set)
    checkpoint_frequency: int = 1  # save checkpoint every N intervals (last interval always saved)
    checkpoint_keep: int = 3  # number of most recent interval checkpoints to keep (0 == keep all)
    seed: int = 42

    wandb: bool = False
    wandb_project: str = 'pixr'
    tensorboard: bool = False
    log_eval_data: bool = False


def _save_checkpoint(trainer: Trainer, checkpoint_dir: str, interval: int, keep: int):
    path = os.path.join(checkpoint_dir, f'checkpoint-{interval}.pt')
    tmp_path = path + '.tmp'
    torch.save(trainer.state_dict(), tmp_path)
    os.replace(tmp_path, path)
    latest = os.path.join(checkpoint_dir, 'checkpoint-latest.pt')
    if os.path.lexists(latest):
        os.remove(latest)
    os.symlink(os.path.basename(path), latest)
    if keep > 0:
        for f in os.listdir(checkpoint_dir):
            stem = f[len('checkpoint-'):-3]
            if f.startswith('checkpoint-') and stem.isdigit() and int(stem) <= interval - keep:
                os.remove(os.path.join(checkpoint_dir, f))
    return path


def main(argv=None):
    train_cfg: TrainCfg = parse_cfg(
        TrainCfg,
        argv=argv,
        task_cfgs=get_task_cfgs(),
        default_task='read',
        description='pixr train',
    )
    task_name = train_cfg.task_name

    device_env = DeviceEnv()
    random_seed(train_cfg.seed, rank=device_env.global_rank)

    if train_cfg.experiment is None:
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        if device_env.world_size > 1:
            date_str = device_env.broadcast_object(date_str)
        experiment = '-'.join([
            date_str,
            f"task_{task_name}",
            f"model_{clean_name(train_cfg.task.model.name)}",
            f"lr_{'{:.1e}'.format(train_cfg.trainer.opt.learning_rate)}",
            f"b_{train_cfg.train_data.batch_size}",
        ])
        train_cfg = replace(train_cfg, experiment=experiment)

    experiment_path = os.path.join(train_cfg.output_dir, train_cfg.experiment)
    checkpoint_dir = os.path.join(experiment_path, 'checkpoints')
    resume = train_cfg.resume
    if resume == 'latest':
        resume = get_latest_checkpoint(checkpoint_dir) if os.path.isdir(checkpoint_dir) else ''
        resume = resume or ''
    log_path = None
    if device_env.is_primary():
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_path = os.path.join(experiment_path, train_cfg.log_filename)
        if os.path.exists(log_path) and not resume:
            print(f"Experiment {train_cfg.experiment} already exists, set --resume or a new --experiment.")
            return -1
        train_cfg.save(os.path.join(experiment_path, 'config.yaml'))

    setup_logging(log_path if device_env.is_primary() else None)
    if not device_env.is_primary():
        logging.root.setLevel(logging.WARNING)
    _logger.info(f"Device env is {device_env}")
    if device_env.is_primary():
        _logger.info('Train config:\n' + train_cfg.dumps_yaml())

    monitor = Monitor(
        train_cfg.experiment,
        output_dir=experiment_path,
        wandb=train_cfg.wandb,
        wandb_project=train_cfg.wandb_project,
        tensorboard=train_cfg.tensorboard,
        output_enabled=device_env.is_primary(),
        log_eval_data=train_cfg.log_eval_data,
        hparams=train_cfg.to_dict(),
    )

    task = get_task(task_name)(train_cfg.task, device_env=device_env, checkpoint_path=train_cfg.checkpoint_path)
    trainer = Trainer(task, train_cfg.trainer, device_env=device_env, monitor=monitor)

    assert train_cfg.train_data.source, 'Train dataset (--train-data.source) must be set.'
    dist_kwargs = dict(
        world_size=device_env.world_size,
        global_rank=device_env.global_rank,
        local_rank=device_env.local_rank,
    )
    train_loader = create_loader(
        train_cfg.train_data, task.train_data_cfg(), is_training=True, seed=train_cfg.seed, **dist_kwargs)
    eval_loader = None
    if train_cfg.eval_data is not None and train_cfg.eval_data.source:
        eval_loader = create_loader(train_cfg.eval_data, task.eval_data_cfg(), is_training=False, **dist_kwargs)

    trainer.setup(
        num_batches_per_interval=train_loader.num_batches,
        resume=resume,
        mean_batch_size=train_loader.mean_batch_size,
    )
    if device_env.is_primary():
        num_params = sum(p.numel() for p in task.trainable_module.parameters())
        _logger.info(f'Model params: {num_params / 1e6:.1f}M, steps per interval: {trainer.num_steps_per_interval}')

    for interval in range(trainer.interval_idx, train_cfg.trainer.num_intervals):
        train_loader.set_interval(interval)
        results = {'train': trainer.train_interval(train_loader)}

        last = interval + 1 == train_cfg.trainer.num_intervals
        if eval_loader is not None and ((interval + 1) % train_cfg.eval_frequency == 0 or last):
            eval_metrics = task.evaluate(eval_loader.loader, use_ema=train_cfg.eval_ema)
            results['eval'] = eval_metrics
            monitor.log_phase('eval', interval=interval, **eval_metrics)

        if device_env.is_primary():
            monitor.write_summary(results, index=interval)
            if (interval + 1) % train_cfg.checkpoint_frequency == 0 or last:
                path = _save_checkpoint(trainer, checkpoint_dir, interval, train_cfg.checkpoint_keep)
                _logger.info(f'Saved checkpoint {path}')


if __name__ == '__main__':
    sys.exit(main())
