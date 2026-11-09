""" Batched greedy generation w/ KV caching for Cruller (image encoder + text decoder) models. """
from typing import Dict, List, Optional, Sequence, Union

import torch

from pixparse.models import Cruller


@torch.no_grad()
def generate_greedy(
        model: Cruller,
        image_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        prompt_ids: Sequence[Sequence[int]],
        stop_token_ids: Sequence[int],
        pad_token_id: int,
        max_new_tokens: int = 128,
) -> List[List[int]]:
    """ Greedy decode a batch of (variable length) prompts conditioned on images.

    Prompts are left padded w/ per-sample position ids so all samples decode in lockstep.

    Returns:
        Generated token ids per sample (excluding prompt), including the stop token if generated.
    """
    decoder = model.text_decoder
    encoder_output, encoder_mask = model.encode(image_input)
    cross_kv = decoder.project_cross_kv(encoder_output)

    device = encoder_output.device
    batch_size = len(prompt_ids)
    prompt_len = max(len(p) for p in prompt_ids)
    max_new_tokens = min(max_new_tokens, decoder.max_length - prompt_len)

    input_ids = torch.full((batch_size, prompt_len), pad_token_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((batch_size, prompt_len), dtype=torch.bool, device=device)
    for i, p in enumerate(prompt_ids):
        input_ids[i, prompt_len - len(p):] = torch.tensor(p, dtype=torch.long, device=device)
        attn_mask[i, prompt_len - len(p):] = True
    position_ids = (attn_mask.long().cumsum(1) - 1).clamp(min=0)

    cache = [{} for _ in decoder.layers]
    logits = decoder(
        input_ids,
        cross_kv=cross_kv,
        encoder_mask=encoder_mask,
        position_ids=position_ids,
        attention_mask=attn_mask,
        cache=cache,
    )
    next_ids = logits[:, -1].argmax(-1)
    stop_ids = torch.tensor(list(stop_token_ids), device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    cur_pos = position_ids[:, -1]
    generated = []
    for _ in range(max_new_tokens):
        next_ids = torch.where(finished, torch.full_like(next_ids, pad_token_id), next_ids)
        generated.append(next_ids)
        finished |= torch.isin(next_ids, stop_ids)
        if finished.all():
            break
        cur_pos = cur_pos + 1
        attn_mask = torch.cat([attn_mask, torch.ones((batch_size, 1), dtype=torch.bool, device=device)], dim=1)
        logits = decoder(
            next_ids[:, None],
            cross_kv=cross_kv,
            encoder_mask=encoder_mask,
            position_ids=cur_pos[:, None],
            attention_mask=attn_mask,
            cache=cache,
        )
        next_ids = logits[:, -1].argmax(-1)

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
