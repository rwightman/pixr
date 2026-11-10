""" Batched greedy / beam search generation w/ KV caching for pixr image -> text models.

Models implement a small incremental decoding interface:
    encode_image(image_input) -> memory (per image, model specific, computed once)
    prefill(memory, prompt_ids, prompt_mask) -> (next token logits [N, V], state)
    step(token_ids [N, 1], state) -> (next token logits [N, V], state)
    state.reorder(parent_idx) selects / duplicates hypotheses (see pixr.models.kv_cache.DecodeState)
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
        model: model implementing encode_image() / prefill() / step()
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

    memory = model.encode_image(image_input)
    logits, state = model.prefill(memory, ids, mask)
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
    return _truncate_at_stop(torch.stack(generated, dim=1).tolist(), stop_token_ids)


def _truncate_at_stop(sequences: List[List[int]], stop_token_ids: Sequence[int]) -> List[List[int]]:
    stop_set = set(stop_token_ids)
    outputs = []
    for seq in sequences:
        out = []
        for t in seq:
            out.append(t)
            if t in stop_set:
                break
        outputs.append(out)
    return outputs


@torch.no_grad()
def generate_beam(
        model,
        image_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        prompt_ids: Sequence[Sequence[int]],
        stop_token_ids: Sequence[int],
        pad_token_id: int,
        max_new_tokens: int = 128,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        bad_token_ids: Optional[Sequence[int]] = None,
        return_scores: bool = False,
):
    """ Beam search decode a batch of (variable length) prompts conditioned on images.

    The image is encoded and the prompt run once per sample, the sequence state is then expanded to `num_beams`
    hypotheses per sample (image-major: beams of sample i at rows [i * num_beams, (i + 1) * num_beams)) and
    reordered by parent beam every step. Per image memory is never expanded or reordered, beams only move within
    their sample's group, and the batch layout is fixed (samples that are done keep decoding, ignored).

    The beam only holds live (unfinished) hypotheses. A hypothesis that ends w/ a stop token while selected among
    the best `num_beams` candidates of its step moves to a per sample pool of completed hypotheses, where it no
    longer competes w/ shorter (higher raw score) prefixes. The result is the best of the pool by
    sum(log prob) / length ** length_penalty, live beams are added to the pool when the length limit is hit.

    A sample stops early once it has `num_beams` completed hypotheses and its best live beam, ranked at the
    current length, doesn't beat the worst of them. That is a heuristic (as HF generate w/ early_stopping=False),
    not a guarantee that no live beam could still rank higher when length_penalty > 0.

    Args:
        model: model implementing encode_image() / prefill() / step()
        image_input: image tensor or NaFlex patch dict
        prompt_ids: per-sample prompt token ids
        stop_token_ids: a hypothesis is complete after any of these tokens
        pad_token_id: padding token id
        max_new_tokens: max tokens to generate
        num_beams: beams per sample, 1 is equivalent to generate_greedy()
        length_penalty: hypotheses are ranked by sum(log prob) / length ** length_penalty
        bad_token_ids: tokens never generated
        return_scores: also return the sum of token log probs of each returned sequence

    Returns:
        Best generated token ids per sample (excluding prompt), including the stop token if generated.
        W/ return_scores, a (sequences, scores) tuple.
    """
    device = next(model.parameters()).device
    ids, mask = left_pad_prompts(prompt_ids, pad_token_id, device)
    B, prompt_len = ids.shape
    K = num_beams
    max_new_tokens = min(max_new_tokens, model.max_text_length - prompt_len)
    if max_new_tokens <= 0:
        return ([[] for _ in range(B)], [0.] * B) if return_scores else [[] for _ in range(B)]

    bad = torch.tensor(list(bad_token_ids), device=device) if bad_token_ids else None
    stop_ids = torch.tensor(list(stop_token_ids), device=device)
    neg_inf = float('-inf')

    def _log_probs(logits):
        logits = logits.float()
        if bad is not None:
            logits = logits.index_fill(-1, bad, neg_inf)
        return logits.log_softmax(-1)

    def _topk(x: torch.Tensor):
        if K == 1:
            # argmax returns the first maximal index, the same tie handling as generate_greedy()
            idx = x.argmax(dim=-1, keepdim=True)
            return x.gather(-1, idx), idx
        return x.topk(K, dim=-1)

    def _rank(raw: float, length: int) -> float:
        return raw / (length ** length_penalty)

    completed = [[] for _ in range(B)]  # per sample (rank score, raw score, tokens), at most K, best first
    done = [False] * B

    def _add_completed(b: int, raw: float, tokens: List[int]):
        pool = completed[b]
        pool.append((_rank(raw, len(tokens)), raw, tokens))
        pool.sort(key=lambda h: h[0], reverse=True)
        del pool[K:]

    def _select(candidates: torch.Tensor, prefixes: Optional[torch.Tensor], length: int):
        """ candidates [B, P, V] cumulative scores of every continuation of the P live prefixes (`prefixes`
        [B, P, length - 1], None for the first step). Completes hypotheses ending w/ a stop token that rank
        within the K best candidates, returns the K best live continuations (scores, parent, token). """
        P, V = candidates.shape[1:]
        assert K <= V - len(stop_token_ids), 'num_beams must be smaller than the vocab'
        if stop_ids.numel():
            # stop candidates actually selected among the K best (ties resolved by the selection itself, not by
            # score, so one beam picks exactly what greedy's argmax picks)
            top_scores, top_idx = _topk(candidates.reshape(B, P * V))
            top_tokens = top_idx % V
            for b, k in torch.isin(top_tokens, stop_ids).nonzero().tolist():
                if done[b] or top_scores[b, k] == neg_inf:
                    continue
                parent = top_idx[b, k].item() // V
                prefix = prefixes[b, parent].tolist() if prefixes is not None else []
                _add_completed(b, top_scores[b, k].item(), prefix + [top_tokens[b, k].item()])
            candidates = candidates.index_fill(-1, stop_ids, neg_inf)
        scores, idx = _topk(candidates.reshape(B, P * V))
        # Stopping heuristic (as HF w/ early_stopping=False), not a bound: a sample is done once it has K completed
        # hypotheses and its best live beam, ranked at the current length, doesn't beat the worst of them. W/ a
        # positive length_penalty a live beam's normalized score can still improve as it grows, so a hypothesis
        # that would have ranked higher can be missed. Exact for length_penalty <= 0.
        best_live = scores[:, 0].tolist()
        for b in range(B):
            if not done[b] and len(completed[b]) >= K and completed[b][-1][0] >= _rank(best_live[b], length):
                done[b] = True
        return scores, idx // V, idx % V

    # image encoding and the prompt run once per sample, before expanding to beams
    memory = model.encode_image(image_input)
    logits, state = model.prefill(memory, ids, mask)
    scores, _, tokens = _select(_log_probs(logits)[:, None], None, 1)  # [B, K], K best first tokens start the beams
    V = logits.shape[-1]
    group_offset = torch.arange(B, device=device)[:, None] * K
    state.reorder(torch.arange(B, device=device).repeat_interleave(K))  # every beam starts from its sample's prompt
    sequences = tokens[:, :, None]  # [B, K, L] live hypotheses, all of length L

    for length in range(2, max_new_tokens + 1):
        if all(done):
            break
        logits, state = model.step(tokens.reshape(B * K, 1), state)
        candidates = scores[:, :, None] + _log_probs(logits).view(B, K, V)
        scores, parent, tokens = _select(candidates, sequences, length)  # parent beams (within the sample) may repeat
        sequences = torch.cat([
            sequences.gather(1, parent[:, :, None].expand(-1, -1, sequences.shape[2])),
            tokens[:, :, None],
        ], dim=2)
        state.reorder((group_offset + parent).reshape(-1))

    # length limit (or all samples done), unfinished beams of samples that aren't done compete as they are
    live_scores, live_sequences = scores.tolist(), sequences.tolist()
    outputs, output_scores = [], []
    for b in range(B):
        if not done[b]:
            for raw, seq in zip(live_scores[b], live_sequences[b]):
                if raw > neg_inf:
                    _add_completed(b, raw, seq)
        _, raw, seq = completed[b][0]
        outputs.append(seq)
        output_scores.append(raw)
    if return_scores:
        return outputs, output_scores
    return outputs
