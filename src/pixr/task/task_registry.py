from copy import deepcopy

from .task_cruller_eval_docvqa import TaskCrullerEvalDocVqa, TaskCrullerEvalDocVqaCfg
from .task_cruller_eval_ocr import TaskCrullerEvalOcr, TaskCrullerEvalOcrCfg
from .task_cruller_finetune_docvqa import TaskCrullerFinetuneDocVqa, TaskCrullerFinetuneDocVqaCfg
from .task_cruller_pretrain import TaskCrullerPretrain, TaskCrullerPretrainCfg

TRAIN_TASK_REGISTRY = {
    "cruller_pretrain": (TaskCrullerPretrain, TaskCrullerPretrainCfg),
    "cruller_finetune_docvqa": (TaskCrullerFinetuneDocVqa, TaskCrullerFinetuneDocVqaCfg),
}

EVAL_TASK_REGISTRY = {
    "cruller_eval_ocr": (TaskCrullerEvalOcr, TaskCrullerEvalOcrCfg),
    "cruller_eval_docvqa": (TaskCrullerEvalDocVqa, TaskCrullerEvalDocVqaCfg),
}


def get_train_task_cfgs():
    return {k: v[1] for k, v in TRAIN_TASK_REGISTRY.items()}


def get_eval_task_cfgs():
    return {k: v[1] for k, v in EVAL_TASK_REGISTRY.items()}


def get_train_task(name):
    return TRAIN_TASK_REGISTRY[name][0]


def get_eval_task(name):
    return EVAL_TASK_REGISTRY[name][0]
