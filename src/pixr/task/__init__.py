from .common import ImageTextTask, init_model_and_tokenizer
from .docvqa import DocVqaTask, DocVqaTaskCfg
from .read import ReadTask, ReadTaskCfg
from .task_registry import TASK_REGISTRY, get_task, get_task_cfgs, get_task_name
