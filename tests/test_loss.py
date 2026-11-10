import pytest
import torch
import torch.nn.functional as F

from pixr.loss import fused_linear_cross_entropy, lm_cross_entropy
from pixr.models import create_model

from test_models import _patches, _tiny_cfg, _tiny_cronut_cfg


def _reference(hidden, weight, bias, target, label_smoothing):
    logits = F.linear(hidden, weight, bias).float()
    ce = F.cross_entropy(logits, target, ignore_index=-100, label_smoothing=label_smoothing)
    valid = target != -100
    z = torch.logsumexp(logits, -1)[valid].square().mean()
    return ce, z


@pytest.mark.parametrize('z_loss', [False, True])
@pytest.mark.parametrize('label_smoothing', [0., 0.1])
@pytest.mark.parametrize('use_bias', [False, True])
def test_fused_ce_matches_reference(z_loss, label_smoothing, use_bias):
    torch.manual_seed(0)
    N, D, V = 37, 16, 50
    hidden = torch.randn(N, D, requires_grad=True)
    weight = torch.randn(V, D, requires_grad=True)
    bias = torch.randn(V, requires_grad=True) if use_bias else None
    target = torch.randint(0, V, (N,))
    target[::4] = -100

    def grads(loss):
        params = [hidden, weight] + ([bias] if use_bias else [])
        return torch.autograd.grad(loss, params)

    ref_ce, ref_z = _reference(hidden, weight, bias, target, label_smoothing)
    ref_loss = ref_ce + (1e-2 * ref_z if z_loss else 0)
    ce, z = fused_linear_cross_entropy(
        hidden, weight, target, bias=bias, chunk_size=8, z_loss=z_loss, label_smoothing=label_smoothing)
    assert (z is None) == (not z_loss)
    torch.testing.assert_close(ce, ref_ce)
    if z_loss:
        torch.testing.assert_close(z, ref_z)
    loss = ce + (1e-2 * z if z_loss else 0)
    for g, rg in zip(grads(loss), grads(ref_loss)):
        torch.testing.assert_close(g, rg)

    # materialized logits path
    ce2, z2 = lm_cross_entropy(
        F.linear(hidden, weight, bias).reshape(1, N, V), target[None], z_loss=z_loss, label_smoothing=label_smoothing)
    torch.testing.assert_close(ce2, ref_ce)
    if z_loss:
        torch.testing.assert_close(z2, ref_z)


def test_fused_ce_no_valid_tokens():
    hidden = torch.randn(4, 8, requires_grad=True)
    weight = torch.randn(10, 8, requires_grad=True)
    ce, z = fused_linear_cross_entropy(hidden, weight, torch.full((4,), -100), z_loss=True)
    (ce + z).backward()
    assert ce.item() == 0 and weight.grad is not None and hidden.grad is not None


@pytest.mark.parametrize('model_type', ['cruller', 'fritter', 'cronut'])
def test_return_hidden_matches_logits(model_type):
    torch.manual_seed(0)
    cfg = _tiny_cronut_cfg() if model_type == 'cronut' else _tiny_cfg(model_type)
    model = create_model(cfg, vocab_size=100, pad_token_id=1).eval()
    patches = _patches()
    text = torch.randint(3, 100, (2, 7))
    with torch.no_grad():
        logits = model(patches, text)
        hidden = model(patches, text, return_hidden=True)
    weight, bias = model.lm_head_params()
    torch.testing.assert_close(F.linear(hidden, weight, bias), logits)
