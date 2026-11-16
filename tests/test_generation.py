import random

import pytest
import torch

from pixr.generation import RepeatStopCfg, _loop_period, generate_greedy, trim_repeat_loop
from pixr.task.common import noise_text_input

EOS = 2
PAD = 0


def _cfg(**kwargs):
    return RepeatStopCfg(**{**dict(min_repeats=8, min_span=32, max_period=16), **kwargs})


def test_trim_repeat_loop():
    cfg = _cfg()
    head = [10, 11, 12, 13]
    unit = [20, 21, 22]
    seq = head + unit * 20 + unit[:1]  # loop running to the length limit, ends mid unit
    out, trimmed = trim_repeat_loop(seq, cfg)
    assert trimmed and out == head + unit
    out, _ = trim_repeat_loop(seq, _cfg(keep=2))
    assert out == head + unit * 2
    # ended by the model (stop token): left as is
    out, trimmed = trim_repeat_loop(seq + [EOS], cfg, stop_token_ids=[EOS])
    assert not trimmed and out == seq + [EOS]
    # short runs below min_span / min_repeats aren't loops
    short = head + [7] * 20 + [8, 9]
    assert trim_repeat_loop(short, cfg) == (short, False)
    assert trim_repeat_loop(head + unit * 7, cfg) == (head + unit * 7, False)
    # single token loop needs min_span copies
    assert trim_repeat_loop(head + [5] * 32, cfg) == (head + [5], True)
    assert trim_repeat_loop(head + [5] * 31, cfg)[1] is False
    # disabled
    assert trim_repeat_loop(seq, RepeatStopCfg()) == (seq, False)


def test_loop_period_matches_trim():
    rng = random.Random(0)
    cfg = _cfg()
    seqs = []
    for i in range(64):
        n = rng.randint(1, 80)
        s = [rng.randint(3, 9) for _ in range(n)]
        if i % 2:
            unit = [rng.randint(3, 9) for _ in range(rng.randint(1, 16))]
            s += unit * rng.randint(1, 40)
        seqs.append(s[:200])
    L = max(len(s) for s in seqs)
    for length in (40, 120, L):
        rows = [s[:length] for s in seqs]
        tokens = torch.tensor([r + [PAD] * (length - len(r)) for r in rows])
        period = _loop_period(tokens, length, cfg)
        for r, p in zip(rows, period.tolist()):
            if len(r) < length:
                continue  # padded rows (finished) are excluded by the caller
            assert (p > 0) == trim_repeat_loop(r, cfg)[1]


class _ScriptedModel(torch.nn.Module):
    """ Emits a fixed token script per row (argmax of one-hot logits), ignoring images / state. """

    def __init__(self, scripts, vocab=32):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.scripts = scripts
        self.vocab = vocab
        self.max_text_length = 1024

    def encode_image(self, image_input):
        return None

    def _logits(self, t):
        ids = torch.tensor([s[min(t, len(s) - 1)] for s in self.scripts])
        return torch.nn.functional.one_hot(ids, self.vocab).float()

    def prefill(self, memory, ids, mask):
        return self._logits(0), 1

    def step(self, token_ids, t):
        return self._logits(t), t + 1


def test_greedy_repeat_stop():
    head = [10, 11, 12, 13, 14]
    looping = head + [20, 21] * 300
    ending = head + [15, 16, EOS]
    model = _ScriptedModel([looping, ending])
    kwargs = dict(prompt_ids=[[1], [1]], stop_token_ids=[EOS], pad_token_id=PAD, max_new_tokens=500)
    plain = generate_greedy(model, None, **kwargs)
    assert len(plain[0]) == 500 and plain[1] == ending
    cfg = _cfg(check_every=8)
    stopped = generate_greedy(model, None, repeat_stop=cfg, **kwargs)
    # stopped within check_every steps of the loop becoming detectable (5 head + 32 span tokens)
    assert len(stopped[0]) <= len(head) + cfg.span(2) + cfg.check_every
    assert stopped[1] == ending
    assert trim_repeat_loop(stopped[0], cfg, [EOS]) == (head + [20, 21], True)


def test_noise_text_input():
    g = torch.Generator().manual_seed(0)
    B, T = 8, 512
    text_input = torch.randint(3, 1000, (B, T), generator=g)
    text_input[:, 0] = 1  # prompt
    text_input[:, 400:] = PAD
    text_target = text_input.clone()
    text_target[:, 0] = -100
    text_target[:, 400:] = -100
    noised = noise_text_input(text_input, text_target, 0.1, generator=g)
    content = text_target != -100
    changed = noised != text_input
    assert not changed[~content].any()  # prompt / pad untouched
    assert 0.07 < changed[content].float().mean() < 0.11  # ~prob, minus same token draws
    assert set(noised[changed].tolist()) <= set(text_input[content].tolist())  # in-batch substitutes
    assert torch.equal(noise_text_input(text_input, text_target, 0.), text_input)
