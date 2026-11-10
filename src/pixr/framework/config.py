from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

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
    z_loss_weight: float = 0.  # weight of the LM z-loss (mean logsumexp(logits) ** 2), e.g. 1e-4
    # train text is trimmed to the longest sequence of the batch rounded up to this multiple, bounds the number of
    # distinct text lengths (compile / FlexAttention mask shapes) for a few extra pad tokens, 1 disables rounding
    text_pad_multiple: int = 64
    loss_chunk_size: int = 4096  # fused (chunked) LM head + CE, tokens per chunk, 0 to materialize full logits


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
    # per component LR multipliers (components defined by the model's param_layer_map(): 'encoder' == image
    # encoder, 'bridge' == cross-attention / prefix projector, 'decoder' == text decoder / LM trunk)
    encoder_lr_scale: float = 1.0
    bridge_lr_scale: float = 1.0
    decoder_lr_scale: float = 1.0
    # layer-wise LR decay within the encoder / decoder (scale = decay ** (top layer - layer id)), None disables
    encoder_layer_decay: Optional[float] = None
    decoder_layer_decay: Optional[float] = None
    # extra multipliers by param name regex (re.search, first match wins), e.g. {'embed_tokens': 2.0}
    lr_scale_overrides: Optional[Dict[str, float]] = None
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
    ema_decay: Optional[float] = None  # enable EMA of model weights w/ this decay

    # torch.compile, strategy (as per OpenCLIP --torchcompile-strategy):
    #   'task': compile the task training forward (model + loss), after distributed wrapping
    #   'model': compile the model before DDP wrapping
    #   'blocks': compile blocks in place before DDP wrapping, grad checkpoint recompute stays in eager autograd
    #     (bounds compiled backward memory when whole-graph compile schedules checkpointed recomputes poorly)
    # w/ FSDP, blocks are always compiled per block inside prepare_fsdp() (compile -> composable AC -> FSDP)
    compile: bool = False
    compile_strategy: str = 'task'
    compile_backend: Optional[str] = None
    compile_mode: Optional[str] = None
    compile_dynamic: Optional[bool] = None

    # FSDP2 (fully_shard) instead of DDP for distributed training (params / grads / optimizer state sharded)
    fsdp: bool = False
    fsdp_reshard_after_forward: bool = True  # False keeps block params gathered from forward to backward
    fsdp_offload_cpu: bool = False
