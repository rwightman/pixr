""" pixr evaluation app

Example (DocVQA validation ANLS):
    python -m pixr.app.eval \\
        --task cruller_eval_docvqa \\
        --eval-data.source pixparse/docvqa-single-page-questions --eval-data.split validation \\
        --eval-data.format hfds --eval-data.batch-size 32 \\
        --checkpoint-path output/<experiment>/checkpoints/checkpoint-latest.pt
"""
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from chugr.common import DataArg, Serializable, parse_args

from pixr.data import create_loader
from pixr.framework import DeviceEnv, setup_logging, random_seed
from pixr.task import get_eval_task, get_eval_task_cfgs

_logger = logging.getLogger('eval')


@dataclass
class EvalCfg(Serializable):
    eval_data: DataArg = field(default_factory=lambda: DataArg(source=''))
    task: Serializable = None  # selected task config (subgroup), set via --task <name>
    checkpoint_path: str = ''
    output_dir: Optional[str] = None  # default: <checkpoint dir>/eval
    seed: int = 42


def main(argv=None):
    eval_cfg: EvalCfg = parse_args(
        EvalCfg,
        argv=argv,
        subgroups={'task': get_eval_task_cfgs()},
        subgroup_defaults={'task': 'cruller_eval_docvqa'},
        description='pixr eval',
    )
    task_name = next(k for k, v in get_eval_task_cfgs().items() if isinstance(eval_cfg.task, v))
    task_cls = get_eval_task(task_name)

    device_env = DeviceEnv()
    random_seed(eval_cfg.seed, rank=device_env.global_rank)
    setup_logging(None)
    if device_env.is_primary():
        _logger.info('Eval config:\n' + eval_cfg.dumps_yaml())

    task = task_cls(eval_cfg.task, device_env=device_env, checkpoint_path=eval_cfg.checkpoint_path)
    task.setup()
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
        output_dir = eval_cfg.output_dir or os.path.join(os.path.dirname(os.path.abspath(eval_cfg.checkpoint_path)), 'eval')
        split = eval_cfg.eval_data.split or 'eval'
        name = f'{task_name}-{split}'
        if hasattr(task, 'save_results'):
            task.save_results(output_dir, name=name)
        with open(os.path.join(output_dir, f'{name}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        _logger.info(f'Results written to {output_dir}')


if __name__ == '__main__':
    sys.exit(main())
