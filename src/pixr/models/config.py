import copy
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from chugr.common import Serializable

from pixr.utils.name_utils import _natural_key, clean_name

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"configs/"]
_MODEL_CONFIGS = {}  # model_name: config


_logger = logging.getLogger(__name__)


@dataclass
class ImageEncoderCfg(Serializable):
    """ timm image encoder config.

    Supports NaFlexVit (native aspect / variable resolution, patch sequence input), and fixed size
    timm models (ViT, hybrid ViT, convnets, etc), any model w/ NLC, NCHW, or NHWC feature output.
    """
    name: str = 'vit_base_patch16_224'
    pretrained: bool = True
    # Initialize from the pretrained weights of a different (but weight compatible) timm model. e.g. a fixed
    # resolution CLIP ViT -> NaFlexVit w/ same block structure. Empty == use `name` pretrained weights.
    pretrained_from: str = ''
    model_kwargs: Dict[str, Any] = field(default_factory=dict)  # extra timm.create_model kwargs
    head_type: Optional[str] = None  # output projection ('linear', 'mlp'), added automatically if dims differ
    out_dim: Optional[int] = None  # override output projection dim (default == text decoder dim)
    image_fmt: str = 'L'
    image_size: Optional[Tuple[int, int]] = None  # (H, W) for fixed size models
    max_seq_len: Optional[int] = None  # max patch sequence length for NaFlex models
    image_mean: Optional[Tuple[float, ...]] = None  # override model pretrained cfg mean
    image_std: Optional[Tuple[float, ...]] = None  # override model pretrained cfg std
    drop_rate: Optional[float] = None
    drop_path_rate: Optional[float] = None
    patch_drop_rate: Optional[float] = None

    @property
    def is_naflex(self) -> bool:
        return self.name.startswith('naflexvit')


@dataclass
class TextDecoderCfg(Serializable):
    """ Transformer text decoder config.

    Types:
        'bart': BART style post-norm decoder w/ learned positions (e.g. facebook/bart-base)
        'ettin': ModernBERT style pre-norm decoder w/ RoPE, GeGLU, local/global attn (e.g. jhu-clsp/ettin-decoder-150m)
        'qwen3': Qwen3 decoder-only LM w/ GQA, qk-norm, SwiGLU, RoPE (e.g. Qwen/Qwen3-0.6B-Base), optional cross-attn

    When `pretrained` is set, architecture hparams are taken from the Hugging Face model `name` and weights
    are loaded from it. Otherwise the arch fields below define a randomly initialized decoder.
    """
    type: str = 'bart'
    name: str = 'facebook/bart-base'
    pretrained: bool = True
    num_layers: Optional[int] = 4  # truncate (keep first N) pretrained layers, or set depth if not pretrained
    max_length: int = 1024
    qk_norm_cross: bool = False
    # Re-initialize cross-attention (w/ zero init output projection) instead of using pretrained (BART) weights.
    # Pretrained cross-attn weights expect the pretrained text encoder's features and significantly slow learning
    # w/ an image encoder, zero init output projections start the decoder as the pretrained LM.
    reinit_cross_attn: bool = True
    scale_embedding: Optional[bool] = None  # None == use pretrained config value (False for BART, True for mBART)
    dim: int = 768
    num_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1
    attn_dropout: float = 0.0
    act_dropout: float = 0.0
    # ettin (ModernBERT) / qwen3 specific
    num_kv_heads: Optional[int] = None  # GQA kv heads (qwen3)
    head_dim: Optional[int] = None  # (qwen3)
    cross_attn_every: int = 1  # add cross-attention to every N-th layer (ettin, qwen3), <= 0 disables
    norm_eps: float = 1e-5
    rope_theta: float = 160000.
    sliding_window: int = 64
    global_attn_every: int = 3


@dataclass
class ModelCfg(Serializable):
    """ Model config.

    Types:
        'cruller': image encoder + text decoder w/ cross-attention (Donut style)
        'fritter': image encoder + projector -> image token prefix for a decoder-only LM (prefix-LM attention)
        'cronut': native decoder-only image + text transformer (placeholder, not implemented yet)
    """
    type: str = 'cruller'
    image_encoder: ImageEncoderCfg = field(default_factory=ImageEncoderCfg)
    text_decoder: TextDecoderCfg = field(default_factory=TextDecoderCfg)
    # fritter (prefix) specific
    prefix_token_merge: int = 1  # merge m x m patch tokens before projection (reduces prefix length by m^2)
    projector_type: str = 'mlp'  # image -> text embedding projector ('mlp', 'linear')


@dataclass
class ModelArgs:
    """ Model selection + common overrides from the command line. """
    # specify a preset model configuration to load
    name: str = "cruller_base"

    # common overrides
    image_fmt: Optional[str] = None
    image_size: Optional[Tuple[int, int]] = None
    max_seq_len: Optional[int] = None
    text_max_length: Optional[int] = None
    text_num_layers: Optional[int] = None
    drop_path_rate: Optional[float] = None
    pretrained: Optional[bool] = None  # override pretrained (init from hub weights) for both towers

    def resolve(self) -> ModelCfg:
        """ Load the named model config and apply any overrides. """
        model_cfg = get_model_config(self.name)
        if model_cfg is None:
            raise RuntimeError(f"Model config for {self.name} was not found. Available: {list_models()}")
        return self.apply_overrides(model_cfg)

    def apply_overrides(self, model_cfg: ModelCfg, from_checkpoint: bool = False) -> ModelCfg:
        """ Apply overrides to a model config.

        Args:
            model_cfg: base model config
            from_checkpoint: weights will be loaded from a pixr checkpoint, disable hub pretrained weights
        """
        enc_updates = {}
        if self.image_size is not None:
            enc_updates['image_size'] = tuple(self.image_size)
        if self.image_fmt is not None:
            enc_updates['image_fmt'] = self.image_fmt
        if self.max_seq_len is not None:
            enc_updates['max_seq_len'] = self.max_seq_len
        if self.drop_path_rate is not None:
            enc_updates['drop_path_rate'] = self.drop_path_rate
        dec_updates = {}
        if self.text_max_length is not None:
            dec_updates['max_length'] = self.text_max_length
        if self.text_num_layers is not None:
            dec_updates['num_layers'] = self.text_num_layers
        if from_checkpoint:
            enc_updates['pretrained'] = False
            dec_updates['pretrained'] = False
        elif self.pretrained is not None:
            enc_updates['pretrained'] = self.pretrained
            dec_updates['pretrained'] = self.pretrained

        return replace(
            model_cfg,
            image_encoder=replace(model_cfg.image_encoder, **enc_updates),
            text_decoder=replace(model_cfg.text_decoder, **dec_updates),
        )


def _scan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        model_cfg = ModelCfg.load(cf)
        _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_scan_model_configs()  # initial populate of model config registry


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def get_model_config(model_name):
    model_name = clean_name(model_name)
    cfg = _MODEL_CONFIGS.get(model_name, None)
    return copy.deepcopy(cfg)
