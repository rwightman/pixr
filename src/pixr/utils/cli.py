""" Command line / config file parsing for pixr apps (jsonargparse based).

* nested dataclass fields map to dotted args, written w/ dashes: `--trainer.opt.learning-rate 1e-4`
  (the underscore form `--trainer.opt.learning_rate` is accepted too)
* `--task <name>` selects the task config dataclass from a registry, `task_name` in a `--config` yaml works too
* jsonargparse provides `--config file.yaml` (CLI args override), `--print-config`, and type validation,
  sequences can be passed as `[1024, 2048]`
"""
import argparse
import dataclasses
import re
import sys
from typing import Dict, Optional, Sequence, Type, TypeVar

import jsonargparse
import yaml

T = TypeVar('T')

_OPT_RE = re.compile(r'--[A-Za-z0-9_.\-]+')


def _normalize_arg(arg: str) -> str:
    """ `--a.b-c=1` -> `--a.b_c=1` (option names only, values untouched). """
    if not arg.startswith('--'):
        return arg
    name, sep, value = arg.partition('=')
    return '--' + name[2:].replace('-', '_') + sep + value


class _DashHelpParser(jsonargparse.ArgumentParser):
    """ Displays option names w/ dashes in help / usage. """

    def format_help(self) -> str:
        return _OPT_RE.sub(lambda m: '--' + m.group(0)[2:].replace('_', '-'), super().format_help())

    def format_usage(self) -> str:
        return _OPT_RE.sub(lambda m: '--' + m.group(0)[2:].replace('_', '-'), super().format_usage())


def _pop_task(argv: Sequence[str]):
    """ Remove `--task <name>` / `--task=<name>` (but not `--task.xxx`) from argv, return (name, remaining). """
    out = []
    name = None
    it = iter(range(len(argv)))
    skip = False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a == '--task':
            name = argv[i + 1]
            skip = True
        elif a.startswith('--task='):
            name = a.split('=', 1)[1]
        else:
            out.append(a)
    return name, out


def _config_task_name(argv: Sequence[str]) -> Optional[str]:
    for i, a in enumerate(argv):
        path = None
        if a == '--config' and i + 1 < len(argv):
            path = argv[i + 1]
        elif a.startswith('--config='):
            path = a.split('=', 1)[1]
        if path:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            if data.get('task_name'):
                return data['task_name']
    return None


def parse_cfg(
        cfg_cls: Type[T],
        argv: Optional[Sequence[str]] = None,
        task_cfgs: Optional[Dict[str, type]] = None,
        default_task: Optional[str] = None,
        description: Optional[str] = None,
) -> T:
    """ Parse command line (+ optional config file) into `cfg_cls`.

    `cfg_cls` must have `task` and `task_name` fields when `task_cfgs` is given, the `task` field is specialized
    to the selected task config dataclass.
    """
    argv = [_normalize_arg(a) for a in (sys.argv[1:] if argv is None else argv)]
    concrete = cfg_cls
    task_name = None
    if task_cfgs:
        task_name, argv = _pop_task(argv)
        task_name = task_name or _config_task_name(argv) or default_task
        if task_name not in task_cfgs:
            raise SystemExit(f'Unknown or missing --task, choose one of {list(task_cfgs.keys())}.')
        task_cls = task_cfgs[task_name]
        concrete = dataclasses.make_dataclass(
            f'{cfg_cls.__name__}_{task_name}',
            [('task', task_cls, dataclasses.field(default_factory=task_cls))],
            bases=(cfg_cls,),
        )

    parser = _DashHelpParser(description=description, exit_on_error=True)
    parser.add_argument('--config', action=jsonargparse.ActionConfigFile, help='yaml or json config file')
    parser.add_class_arguments(concrete)
    ns = parser.parse_args(argv)
    values = {k: v for k, v in vars(parser.instantiate(ns)).items() if k != 'config' and not k.startswith('__')}
    if task_cfgs:
        values['task_name'] = task_name
    return concrete(**values)
