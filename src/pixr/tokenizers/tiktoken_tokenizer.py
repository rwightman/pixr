""" tiktoken based tokenizer w/ a (minimal) Hugging Face tokenizer compatible interface.

Follows the open_clip GenLIP / ModernText convention: the base BPE encoding (e.g. cl100k_base) handles text, control
tokens (eos, pad, bos) are reserved directly above the base vocab, and added (task) special tokens follow. Special
token strings present in text are matched and mapped to their ids, everything else is encoded w/ encode_ordinary.
"""
import re
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Union

import torch

TIKTOKEN_PREFIX = 'tiktoken:'


class TiktokenTokenizer:
    eos_token = '<|eos|>'
    pad_token = '<|pad|>'
    bos_token = '<|bos|>'

    def __init__(self, encoding_name: str = 'cl100k_base'):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding_name)
        self.name_or_path = TIKTOKEN_PREFIX + encoding_name
        self.vocab_size = self.enc.n_vocab  # base vocab, control & added tokens are above
        self._special: Dict[str, int] = {}
        for t in (self.eos_token, self.pad_token, self.bos_token):
            self._special[t] = self.vocab_size + len(self._special)
        self._id_to_special = {v: k for k, v in self._special.items()}
        # ids below n_vocab that never occur in encoded text, a model can still output them:
        # * ids w/o a token (cl100k_base has 16), decoding one raises
        # * the encoding's own special tokens (<|endoftext|>, ...), text is encoded w/ encode_ordinary (pixr control
        #   tokens are above the base vocab), decoding one would produce its literal string
        self._unused_ids = {self.enc.encode_single_token(t) for t in self.enc.special_tokens_set}
        for i in range(self.vocab_size):
            try:
                self.enc.decode_single_token_bytes(i)
            except KeyError:
                self._unused_ids.add(i)
        self._split_re = None
        self._update_split()

    @property
    def eos_token_id(self) -> int:
        return self._special[self.eos_token]

    @property
    def pad_token_id(self) -> int:
        return self._special[self.pad_token]

    @property
    def bos_token_id(self) -> int:
        return self._special[self.bos_token]

    def __len__(self):
        return self.vocab_size + len(self._special)

    def _update_split(self):
        tokens = sorted(self._special.keys(), key=len, reverse=True)
        self._split_re = re.compile('(' + '|'.join(re.escape(t) for t in tokens) + ')')

    def add_tokens(self, tokens: Sequence[str], special_tokens: bool = True) -> int:
        added = 0
        for t in tokens:
            if t not in self._special:
                self._special[t] = self.vocab_size + len(self._special)
                self._id_to_special[self._special[t]] = t
                added += 1
        if added:
            self._update_split()
        return added

    @property
    def unused_token_ids(self) -> List[int]:
        """ Base vocab ids that never occur in encoded text (no token, or the encoding's own special tokens), to be
        suppressed in generation (ignored by decode). """
        return sorted(self._unused_ids)

    @property
    def all_special_ids(self) -> List[int]:
        # control (eos / pad / bos) and added tokens, all special
        return list(self._special.values())

    def get_added_vocab(self) -> Dict[str, int]:
        # tokens above the base BPE vocab (control & added), as HF tokenizers
        return dict(self._special)

    def get_vocab(self) -> Dict[str, int]:
        # NOTE only special / added tokens, the base BPE vocab is bytes based (not needed by pixr)
        return dict(self._special)

    def convert_tokens_to_ids(self, tokens: Union[str, Sequence[str]]):
        if isinstance(tokens, str):
            return self._special[tokens]
        return [self._special[t] for t in tokens]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = []
        for piece in self._split_re.split(text):
            if not piece:
                continue
            if piece in self._special:
                ids.append(self._special[piece])
            else:
                ids.extend(self.enc.encode_ordinary(piece))
        return ids

    def __call__(
            self,
            text: Union[str, Sequence[str]],
            add_special_tokens: bool = False,
            return_tensors: Optional[str] = None,
            max_length: Optional[int] = None,
            padding: Union[bool, str] = False,
            truncation: bool = False,
    ):
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        batch = [self.encode(t) for t in texts]
        if truncation and max_length:
            batch = [ids[:max_length] for ids in batch]
        if padding == 'max_length' and max_length:
            batch = [ids + [self.pad_token_id] * (max_length - len(ids)) for ids in batch]
        elif padding in (True, 'longest'):
            longest = max(len(ids) for ids in batch)
            batch = [ids + [self.pad_token_id] * (longest - len(ids)) for ids in batch]
        if return_tensors == 'pt':
            input_ids = torch.tensor(batch, dtype=torch.long)
        else:
            input_ids = batch[0] if single else batch
        return SimpleNamespace(input_ids=input_ids)

    def decode(self, ids: Union[Sequence[int], torch.Tensor], skip_special_tokens: bool = False) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        out, run = [], []
        for i in ids:
            if i < self.vocab_size:
                if i not in self._unused_ids:  # no token (enc.decode would raise) or a base special token
                    run.append(i)
                continue
            if run:
                out.append(self.enc.decode(run))
                run = []
            if not skip_special_tokens and i in self._id_to_special:
                out.append(self._id_to_special[i])
        if run:
            out.append(self.enc.decode(run))
        return ''.join(out)

    def batch_decode(self, batch, skip_special_tokens: bool = False) -> List[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch]
