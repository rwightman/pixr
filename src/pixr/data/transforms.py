from typing import Any, Dict, Optional

import chugr
from chugr.common import ImageInputCfg


def create_image_preprocessor(
        input_info: Dict[str, Any],
        is_training: bool,
        transform_type: Optional[str] = None,
        max_seq_len: Optional[int] = None,
):
    """ Create image preprocessing for a model's image input requirements (see ImageEncoderTimm.input_info).

    The same transform type is used for train and eval (eval disables augmentation), so input geometry
    (resize, padding, patch sequence) always matches between training and evaluation.

    Args:
        input_info: model image input info
        is_training: enable train augmentations
        transform_type: chugr transform type, default 'doc_naflex' for NaFlex models, else 'doc_better'
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
        max_seq_len=max_seq_len or input_info['max_seq_len'],
    )
    return chugr.create_image_preprocessor(cfg, is_training=is_training)


class ImagePreprocessorFactory:
    """ seq_len -> image preprocessor, for NaFlex bucketed batching (picklable for dataloader workers). """

    def __init__(self, input_info: Dict[str, Any], is_training: bool, transform_type: Optional[str] = None):
        self.input_info = dict(input_info)
        self.is_training = is_training
        self.transform_type = transform_type

    def __call__(self, seq_len: int):
        return create_image_preprocessor(
            self.input_info, self.is_training, transform_type=self.transform_type, max_seq_len=seq_len)
