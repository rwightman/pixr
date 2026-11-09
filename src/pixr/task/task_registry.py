from .docvqa import DocVqaTask, DocVqaTaskCfg
from .read import ReadTask, ReadTaskCfg

TASK_REGISTRY = {
    'read': (ReadTask, ReadTaskCfg),
    'docvqa': (DocVqaTask, DocVqaTaskCfg),
}


def get_task_cfgs():
    return {k: v[1] for k, v in TASK_REGISTRY.items()}


def get_task(name):
    return TASK_REGISTRY[name][0]


def get_task_name(cfg) -> str:
    return next(k for k, v in TASK_REGISTRY.items() if type(cfg) is v[1])
