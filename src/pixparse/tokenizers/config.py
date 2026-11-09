from dataclasses import dataclass

from chug.common import Serializable


@dataclass
class TokenizerCfg(Serializable):
    name: str = 'facebook/bart-base'
