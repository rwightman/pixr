from dataclasses import dataclass, field
from typing import Optional, Tuple

from pixr.models import ModelArgs
from pixr.tokenizers import TokenizerCfg


@dataclass
class OptimizationCfg:
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    learning_rate: float = 1e-4
    warmup_learning_rate: float = 0.
    min_learning_rate: float = 0.
    weight_decay: float = .05
    eps: float = 1e-8
    clip_grad_value: Optional[float] = 1.0  # Gradient clipping value.
    clip_grad_mode: Optional[str] = 'norm'  # Gradient clipping mode, one of ('norm', 'value', 'agc').
    grad_accum_steps: int = 1
    grad_checkpointing: bool = False
    momentum: Optional[float] = None
    betas: Optional[Tuple[float, float]] = (0.9, 0.98)
    layer_decay: Optional[float] = None
    encoder_lr_scale: float = 1.0  # LR multiplier for the (pretrained) image encoder
    label_smoothing: float = 0.


@dataclass
class TaskCfg:
    dtype: Optional[str] = 'bfloat16'  # low precision dtype for autocast ('bfloat16' or 'float16')
    amp: bool = True
    model: ModelArgs = field(default_factory=ModelArgs)
    tokenizer: Optional[TokenizerCfg] = None  # default == text decoder name
    image_transforms: Optional[str] = None  # chugr transform type, default 'doc_naflex' or 'doc_better'


@dataclass
class TaskTrainCfg(TaskCfg):
    num_intervals: int = 100
    num_warmup_intervals: float = 1.
    log_frequency: int = 100  # log every n steps
    metrics_frequency: int = 1000  # calculate (expensive) train metrics every n steps
    opt: OptimizationCfg = field(default_factory=OptimizationCfg)
    compile: bool = False


@dataclass
class TaskEvalCfg(TaskCfg):
    pass
