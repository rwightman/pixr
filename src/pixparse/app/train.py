""" pixparse training app

Example (pretrain on IDL streamed from the HF Hub, 2 GPUs):
    torchrun --nproc_per_node 2 -m pixparse.app.train \\
        --task cruller_pretrain \\
        --task.model.name cruller_naflex_clip_base \\
        --train-data.source "pipe:curl -s -f -L https://huggingface.co/datasets/pixparse/idl-wds/resolve/main/idl-train-0{0000..2999}.tar" \\
        --train-data.num-samples 1000000 --train-data.batch-size 16 --train-data.num-workers 16
"""
import logging
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Optional

import torch

from chug.common import DataArg, Serializable, parse_args

from pixparse.data import create_loader
from pixparse.framework import DeviceEnv, Monitor, TaskTrain, train_one_interval, setup_logging, random_seed
from pixparse.task import get_train_task, get_train_task_cfgs
from pixparse.utils import clean_name, get_latest_checkpoint

_logger = logging.getLogger('train')


@dataclass
class TrainCfg(Serializable):
    train_data: DataArg = field(default_factory=lambda: DataArg(source=''))
    eval_data: Optional[DataArg] = None
    task: Serializable = None  # selected task config (subgroup), set via --task <name>

    experiment: Optional[str] = None  # experiment name, auto-generated if None
    output_dir: str = './output'
    log_filename: str = 'out.log'
    resume: str = ''  # resume training from checkpoint w/ full model + optimizer state ('latest' for most recent)
    checkpoint_path: str = ''  # initialize model weights (and tokens / arch) from this checkpoint
    eval_frequency: int = 1  # evaluate every N intervals (if eval_data set)
    checkpoint_frequency: int = 1  # save checkpoint every N intervals (last interval always saved)
    checkpoint_keep: int = 3  # number of most recent interval checkpoints to keep (0 == keep all)
    seed: int = 42

    wandb: bool = False
    wandb_project: str = 'pixparse'
    tensorboard: bool = False
    log_eval_data: bool = False


def _save_checkpoint(task, checkpoint_dir, interval, keep):
    path = os.path.join(checkpoint_dir, f'checkpoint-{interval}.pt')
    tmp_path = path + '.tmp'
    torch.save(task.state_dict(), tmp_path)
    os.replace(tmp_path, path)
    latest = os.path.join(checkpoint_dir, 'checkpoint-latest.pt')
    if os.path.lexists(latest):
        os.remove(latest)
    os.symlink(os.path.basename(path), latest)
    if keep > 0:
        stale = interval - keep * 1
        for f in os.listdir(checkpoint_dir):
            if f.startswith('checkpoint-') and f[len('checkpoint-'):-3].isdigit():
                if int(f[len('checkpoint-'):-3]) <= stale:
                    os.remove(os.path.join(checkpoint_dir, f))
    return path


def train(cfg: TrainCfg, task: TaskTrain, loaders, checkpoint_dir: str):
    device_env = task.device_env
    train_loader = loaders['train']
    eval_loader = loaders.get('eval', None)
    for i in range(task.interval_idx, task.num_intervals):
        train_loader.set_interval(i)
        train_stats = train_one_interval(task, train_loader)
        results = {'train': train_stats}

        last = i + 1 == task.num_intervals
        if eval_loader is not None and ((i + 1) % cfg.eval_frequency == 0 or last):
            eval_metrics = task.evaluate(eval_loader.loader)
            results['eval'] = eval_metrics
            task.monitor.log_phase('eval', interval=i, **eval_metrics)

        if device_env.is_primary():
            task.monitor.write_summary(results, index=i)
            if (i + 1) % cfg.checkpoint_frequency == 0 or last:
                path = _save_checkpoint(task, checkpoint_dir, i, cfg.checkpoint_keep)
                _logger.info(f'Saved checkpoint {path}')


def main(argv=None):
    train_cfg: TrainCfg = parse_args(
        TrainCfg,
        argv=argv,
        subgroups={'task': get_train_task_cfgs()},
        subgroup_defaults={'task': 'cruller_pretrain'},
        description='pixparse train',
    )
    task_name = next(k for k, v in get_train_task_cfgs().items() if isinstance(train_cfg.task, v))
    task_cls = get_train_task(task_name)

    device_env = DeviceEnv()
    random_seed(train_cfg.seed, rank=device_env.global_rank)

    if train_cfg.experiment is None:
        model_name_safe = clean_name(train_cfg.task.model.name)
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        if device_env.world_size > 1:
            date_str = device_env.broadcast_object(date_str)
        experiment = '-'.join([
            date_str,
            f"task_{task_name}",
            f"model_{model_name_safe}",
            f"lr_{'{:.1e}'.format(train_cfg.task.opt.learning_rate)}",
            f"b_{train_cfg.train_data.batch_size}",
        ])
        train_cfg = replace(train_cfg, experiment=experiment)

    experiment_path = os.path.join(train_cfg.output_dir, train_cfg.experiment)
    checkpoint_dir = os.path.join(experiment_path, 'checkpoints')
    resume = train_cfg.resume
    if resume == 'latest':
        resume = get_latest_checkpoint(checkpoint_dir) or ''
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

    task = task_cls(
        train_cfg.task,
        device_env=device_env,
        monitor=monitor,
        checkpoint_path=train_cfg.checkpoint_path,
    )

    loaders = {}
    assert train_cfg.train_data.source, 'Train dataset (--train-data.source) must be set.'
    loaders['train'] = create_loader(
        train_cfg.train_data,
        task.train_data_cfg(),
        is_training=True,
        seed=train_cfg.seed,
        world_size=device_env.world_size,
        global_rank=device_env.global_rank,
        local_rank=device_env.local_rank,
    )
    if train_cfg.eval_data is not None and train_cfg.eval_data.source:
        assert hasattr(task, 'evaluate'), f'Task {task_name} does not support evaluation.'
        loaders['eval'] = create_loader(
            train_cfg.eval_data,
            task.eval_data_cfg(),
            is_training=False,
            world_size=device_env.world_size,
            global_rank=device_env.global_rank,
            local_rank=device_env.local_rank,
        )

    task.setup(
        num_batches_per_interval=loaders['train'].num_batches,
        resume=resume,
    )
    if resume:
        loaders['train'].set_interval(task.interval_idx)

    if device_env.is_primary():
        num_params = sum(p.numel() for p in task.unwrapped_model.parameters())
        _logger.info(f'Model params: {num_params / 1e6:.1f}M, steps per interval: {task.num_steps_per_interval}')

    train(train_cfg, task, loaders, checkpoint_dir)


if __name__ == '__main__':
    sys.exit(main())
