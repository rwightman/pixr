""" Batched greedy generation w/ KV caching for pixparse image -> text models.

Models implement a small incremental decoding interface:
    prefill(image_input, prompt_ids, prompt_mask) -> (next token logits [B, V], state)
    step(token_ids [B, 1], state) -> (next token logits [B, V], state)
and expose `max_text_length` (max decoder sequence length incl. prompt).

Prompts are left padded (w/ `prompt_mask`) so all samples decode in lockstep.
"""
from typing import Dict, List, Optional, Sequence, Union

import torch


def left_pad_prompts(prompt_ids: Sequence[Sequence[int]], pad_token_id: int, device):
    batch_size = len(prompt_ids)
    prompt_len = max(len(p) for p in prompt_ids)
    ids = torch.full((batch_size, prompt_len), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros((batch_size, prompt_len), dtype=torch.bool, device=device)
    for i, p in enumerate(prompt_ids):
        ids[i, prompt_len - len(p):] = torch.tensor(p, dtype=torch.long, device=device)
        mask[i, prompt_len - len(p):] = True
    return ids, mask


@torch.no_grad()
def generate_greedy(
        model,
        image_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        prompt_ids: Sequence[Sequence[int]],
        stop_token_ids: Sequence[int],
        pad_token_id: int,
        max_new_tokens: int = 128,
        bad_token_ids: Optional[Sequence[int]] = None,
) -> List[List[int]]:
    """ Greedy decode a batch of (variable length) prompts conditioned on images.

    Args:
        model: model implementing prefill() / step()
        image_input: image tensor or NaFlex patch dict
        prompt_ids: per-sample prompt token ids
        stop_token_ids: generation stops (per sample) after any of these tokens
        pad_token_id: padding token id
        max_new_tokens: max tokens to generate
        bad_token_ids: tokens never generated (e.g. <unk>, as per Donut inference)

    Returns:
        Generated token ids per sample (excluding prompt), including the stop token if generated.
    """
    device = next(model.parameters()).device
    ids, mask = left_pad_prompts(prompt_ids, pad_token_id, device)
    batch_size, prompt_len = ids.shape
    max_new_tokens = min(max_new_tokens, model.max_text_length - prompt_len)

    bad = torch.tensor(list(bad_token_ids), device=device) if bad_token_ids else None

    def _select(logits):
        if bad is not None:
            logits = logits.index_fill(-1, bad, float('-inf'))
        return logits.argmax(-1)

    logits, state = model.prefill(image_input, ids, mask)
    next_ids = _select(logits)
    stop_ids = torch.tensor(list(stop_token_ids), device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated = []
    for _ in range(max_new_tokens):
        next_ids = torch.where(finished, torch.full_like(next_ids, pad_token_id), next_ids)
        generated.append(next_ids)
        finished |= torch.isin(next_ids, stop_ids)
        if finished.all():
            break
        logits, state = model.step(next_ids[:, None], state)
        next_ids = _select(logits)

    if not generated:
        return [[] for _ in range(batch_size)]
    generated = torch.stack(generated, dim=1).tolist()
    stop_set = set(stop_token_ids)
    outputs = []
    for seq in generated:
        out = []
        for t in seq:
            out.append(t)
            if t in stop_set:
                break
        outputs.append(out)
    return outputs
