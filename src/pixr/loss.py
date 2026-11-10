""" LM cross-entropy w/ optional z-loss and label smoothing.

`fused_linear_cross_entropy` applies the output projection + CE in (gradient checkpointed) chunks over tokens so the
full [tokens, vocab] logits are never materialized, ported from OpenCLIP (GenLIP caption loss).
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _ce_z_sums(
        logits: torch.Tensor,
        target: torch.Tensor,
        z_loss: bool,
        label_smoothing: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ fp32 CE and squared log normalizer sums over (already filtered, all valid) targets. """
    logits = logits.float()
    if not z_loss:
        ce_sum = F.cross_entropy(logits, target, reduction='sum', label_smoothing=label_smoothing)
        return ce_sum, ce_sum.new_zeros(())
    # CE = logsumexp(logits) - target_logit, reuse log_z for the z-loss
    log_z = torch.logsumexp(logits, dim=-1)
    nll = log_z - logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    if label_smoothing > 0:
        # matches F.cross_entropy(label_smoothing=e): (1 - e) * nll + e * mean over classes of -log p
        nll = (1. - label_smoothing) * nll + label_smoothing * (log_z - logits.mean(dim=-1))
    return nll.sum(), log_z.square().sum()


def _chunk_linear_ce(hidden, weight, bias, target, z_loss: bool, label_smoothing: float):
    # the vocab GEMM stays in the model dtype / ambient autocast, CE + logsumexp in fp32
    return _ce_z_sums(F.linear(hidden, weight, bias), target, z_loss, label_smoothing)


def _graph_tied_zero(hidden, weight, bias):
    """ fp32 zero w/ graph edges to all inputs (keeps backward valid and DDP params used w/ no valid tokens). """
    zero = hidden.sum() + weight[0, 0]
    if bias is not None:
        zero = zero + bias[0]
    return zero.float() * 0.


@torch.compiler.disable
def fused_linear_cross_entropy(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        chunk_size: int = 4096,
        z_loss: bool = False,
        label_smoothing: float = 0.,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """ `cross_entropy(linear(hidden, weight, bias), target)` w/o materializing the full logits.

    Ignored targets are dropped before the head GEMM, the remaining tokens are processed in chunks of `chunk_size`,
    each gradient checkpointed (logits recomputed in backward) so peak memory is one [chunk_size, vocab] block.

    Args:
        hidden: [N, D] pre-head features
        weight: [V, D] output projection
        target: [N] token ids, `ignore_index` positions skipped
        bias: optional [V] output bias
        chunk_size: tokens per chunk
        z_loss: also return the mean squared log normalizer (logsumexp(logits) ** 2)
        label_smoothing: as per F.cross_entropy

    Returns:
        Unweighted (mean CE, mean z) over valid tokens, z is None if `z_loss` is False.
    """
    assert chunk_size > 0
    valid = target != ignore_index
    n_valid = valid.sum()
    hidden = hidden[valid]
    target = target[valid]
    n_tokens = hidden.shape[0]
    use_ckpt = torch.is_grad_enabled() and (
        hidden.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad))

    ce_total = hidden.new_zeros((), dtype=torch.float32)
    z_total = hidden.new_zeros((), dtype=torch.float32)
    if n_tokens == 0 and use_ckpt:
        zero = _graph_tied_zero(hidden, weight, bias)
        ce_total = ce_total + zero
        z_total = z_total + zero
    for start in range(0, n_tokens, chunk_size):
        h_chunk = hidden[start:start + chunk_size]
        t_chunk = target[start:start + chunk_size]
        if use_ckpt:
            ce_chunk, z_chunk = checkpoint(
                _chunk_linear_ce, h_chunk, weight, bias, t_chunk, z_loss, label_smoothing, use_reentrant=False)
        else:
            ce_chunk, z_chunk = _chunk_linear_ce(h_chunk, weight, bias, t_chunk, z_loss, label_smoothing)
        ce_total = ce_total + ce_chunk
        z_total = z_total + z_chunk
    denom = n_valid.clamp(min=1)
    return ce_total / denom, (z_total / denom if z_loss else None)


def lm_cross_entropy(
        logits: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int = -100,
        z_loss: bool = False,
        label_smoothing: float = 0.,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """ Materialized logits counterpart of `fused_linear_cross_entropy`, same return values. """
    logits = logits.flatten(0, -2)
    target = target.flatten()
    valid = target != ignore_index
    n_valid = valid.sum().clamp(min=1)
    logits = logits.float()
    if not z_loss:
        ce = F.cross_entropy(
            logits, target, ignore_index=ignore_index, reduction='sum', label_smoothing=label_smoothing)
        return ce / n_valid, None
    # masked reductions (boolean indexing the logits would save a second [n_valid, vocab] copy for backward)
    log_z = torch.logsumexp(logits, dim=-1)
    safe_target = target.masked_fill(~valid, 0)
    nll = log_z - logits.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    if label_smoothing > 0:
        nll = (1. - label_smoothing) * nll + label_smoothing * (log_z - logits.mean(dim=-1))
    ce = (nll * valid).sum() / n_valid
    z = (log_z.square() * valid).sum() / n_valid
    return ce, z
