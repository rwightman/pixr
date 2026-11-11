""" pixr evaluation app

Example (DocVQA validation ANLS):
    python -m pixr.app.eval \\
        --task docvqa \\
        --eval-data.source pixparse/docvqa-single-page-questions --eval-data.split validation \\
        --eval-data.format hfds --eval-data.batch-size 32 \\
        --checkpoint-path output/<experiment>/checkpoints/checkpoint-latest.pt
"""
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from chugr.common import DataArg, Serializable

from pixr.data import create_loader
from pixr.framework import DeviceEnv, destroy_distributed, setup_logging, random_seed
from pixr.task import get_task, get_task_cfgs
from pixr.utils import load_checkpoint
from pixr.utils.cli import parse_cfg

_logger = logging.getLogger('eval')


@dataclass
class EvalCfg(Serializable):
    eval_data: DataArg = field(default_factory=lambda: DataArg(source=''))
    task_name: str = 'docvqa'  # selected via --task <name> (or task_name in a --config file)
    task: Any = None  # task config, type set by the selected task
    checkpoint_path: str = ''
    use_ema: bool = False  # evaluate EMA weights if present in the checkpoint
    output_dir: Optional[str] = None  # default: <checkpoint dir>/eval
    seed: int = 42


def _run(argv=None):
    eval_cfg: EvalCfg = parse_cfg(
        EvalCfg,
        argv=argv,
        task_cfgs=get_task_cfgs(),
        default_task='docvqa',
        description='pixr eval',
    )
    task_name = eval_cfg.task_name

    device_env = DeviceEnv()
    random_seed(eval_cfg.seed, rank=device_env.global_rank)
    setup_logging(None)
    if device_env.is_primary():
        _logger.info('Eval config:\n' + eval_cfg.dumps_yaml())

    assert eval_cfg.checkpoint_path, 'A checkpoint is required for evaluation (--checkpoint-path).'
    task = get_task(task_name)(eval_cfg.task, device_env=device_env, checkpoint_path=eval_cfg.checkpoint_path)
    if eval_cfg.use_ema:
        checkpoint = load_checkpoint(eval_cfg.checkpoint_path)
        assert 'model_ema' in checkpoint, 'Checkpoint has no EMA weights.'
        task.trainable_module.load_state_dict(checkpoint['model_ema'])
    task.to(device_env.device).eval()

    loader = create_loader(
        eval_cfg.eval_data,
        task.eval_data_cfg(),
        is_training=False,
        world_size=device_env.world_size,
        global_rank=device_env.global_rank,
        local_rank=device_env.local_rank,
    )
    metrics = task.evaluate(loader.loader)

    if device_env.is_primary():
        _logger.info(f'{task_name} results: {metrics}')
        output_dir = eval_cfg.output_dir or os.path.join(
            os.path.dirname(os.path.abspath(eval_cfg.checkpoint_path)), 'eval')
        split = re.sub(r'[^A-Za-z0-9_.+-]+', '_', eval_cfg.eval_data.split or 'eval').strip('_')
        name = f'{task_name}-{split}'
        task.save_results(output_dir, name=name)
        with open(os.path.join(output_dir, f'{name}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        _logger.info(f'Results written to {output_dir}')


def main(argv=None):
    """ Entry point for both launch forms (`pixr-eval` console script, `python -m pixr.app.eval`). """
    code = _run(argv)
    # NOTE only after a normal return, w/ an exception other ranks may still be in collectives, the launcher tears down
    destroy_distributed()
    return code


if __name__ == '__main__':
    sys.exit(main())
