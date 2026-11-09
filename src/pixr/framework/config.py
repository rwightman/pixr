from dataclasses import dataclass, field
from typing import Optional, Tuple

from pixr.models import ModelArgs
from pixr.tokenizers import TokenizerCfg


@dataclass
class TaskCfg:
    """ Base task config: model, tokenizer, preprocessing, and precision (used for training and eval). """
    model: ModelArgs = field(default_factory=ModelArgs)
    tokenizer: Optional[TokenizerCfg] = None  # default == text decoder name (or checkpoint tokenizer)
    image_transforms: Optional[str] = None  # chugr transform type, default 'doc_naflex' or 'doc_better'
    dtype: Optional[str] = 'bfloat16'  # low precision dtype for autocast ('bfloat16' or 'float16')
    amp: bool = True
    label_smoothing: float = 0.


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
    momentum: Optional[float] = None
    betas: Optional[Tuple[float, float]] = (0.9, 0.98)
    layer_decay: Optional[float] = None
    encoder_lr_scale: float = 1.0  # LR multiplier for the (pretrained) image encoder
    # scale loss by batch size relative to the mean for variable batch size (NaFlex schedule) loaders,
    # 'none', 'linear' (per-sample gradient weight constant) or 'sqrt'
    loss_batch_scale: str = 'none'


@dataclass
class TrainerCfg:
    """ Optimization loop config, independent of the task. """
    num_intervals: int = 100
    num_warmup_intervals: float = 1.
    log_frequency: int = 100  # log every n steps
    metrics_frequency: int = 1000  # calculate (expensive) task train metrics every n steps, 0 disables
    opt: OptimizationCfg = field(default_factory=OptimizationCfg)
    grad_checkpointing: bool = False
    compile: bool = False
    ema_decay: Optional[float] = None  # enable EMA of model weights w/ this decay
