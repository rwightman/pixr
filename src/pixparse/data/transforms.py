from typing import Any, Dict, Optional

import chug
from chug.common import ImageInputCfg


def create_image_preprocessor(
        input_info: Dict[str, Any],
        is_training: bool,
        transform_type: Optional[str] = None,
):
    """ Create image preprocessing for a model's image input requirements (see ImageEncoderTimm.input_info).

    The same transform type is used for train and eval (eval disables augmentation), so input geometry
    (resize, padding, patch sequence) always matches between training and evaluation.

    Args:
        input_info: model image input info
        is_training: enable train augmentations
        transform_type: chug transform type, default 'doc_naflex' for NaFlex models, else 'doc_better'
    """
    naflex = input_info['naflex']
    if transform_type is None:
        transform_type = 'doc_naflex' if naflex else 'doc_better'
    if naflex:
        assert transform_type == 'doc_naflex', 'NaFlex models require doc_naflex transforms.'
    cfg = ImageInputCfg(
        size=input_info['image_size'],
        mode=input_info['image_fmt'],
        mean=input_info['mean'],
        std=input_info['std'],
        interpolation='bicubic',
        fill_color=255,
        transform_type=transform_type,
        patch_size=input_info['patch_size'],
        max_seq_len=input_info['max_seq_len'],
    )
    return chug.create_image_preprocessor(cfg, is_training=is_training)
