from dataclasses import dataclass

from chugr.common import Serializable


@dataclass
class TokenizerCfg(Serializable):
    name: str = 'facebook/bart-base'
