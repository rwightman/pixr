from typing import List, Optional, Sequence

from transformers import AutoTokenizer

from .config import TokenizerCfg


def create_tokenizer(cfg: TokenizerCfg, added_tokens: Optional[Sequence[str]] = None):
    """ Create a Hugging Face tokenizer and add special tokens (in order).

    Token order matters, ids are assigned sequentially. When fine-tuning from a checkpoint, pass the
    checkpoint's added tokens first, then task tokens, so existing embedding rows line up.
    """
    assert cfg.name
    tokenizer = AutoTokenizer.from_pretrained(cfg.name)
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        # a pad token distinct from eos is required, padding is excluded from the loss (eos must be learned)
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    if added_tokens:
        add_tokens(tokenizer, added_tokens)
    return tokenizer


def add_tokens(tokenizer, tokens: Sequence[str]) -> int:
    # de-dupe while preserving order, skip tokens already in vocab
    vocab = tokenizer.get_vocab()
    new_tokens = [t for t in dict.fromkeys(tokens) if t not in vocab]
    if not new_tokens:
        return 0
    return tokenizer.add_tokens(new_tokens, special_tokens=True)


def get_added_tokens(tokenizer) -> List[str]:
    """ Tokens added on top of the base vocab, in id order. """
    base_size = tokenizer.vocab_size
    added = {i: t for t, i in tokenizer.get_vocab().items() if i >= base_size}
    return [added[i] for i in sorted(added)]
