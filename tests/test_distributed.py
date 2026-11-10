""" TrainingTask DDP / FSDP2 handling on a single GPU (world size 1 process group).

Multi-GPU parity (loss trajectories, accumulation, compile, resume) is covered by scripts/check_distributed.py.
"""
import copy
import os
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


@pytest.fixture(scope='module')
def process_group():
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29533')
    created = not dist.is_initialized()
    if created:
        dist.init_process_group('nccl', rank=0, world_size=1, device_id=torch.device('cuda', 0))
    # DeviceEnv enables TF32 / cudnn benchmark globally, restore for other tests
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.benchmark
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.benchmark = flags
    if created:
        dist.destroy_process_group()


def _make_task(model_name):
    from pixr.framework import DeviceEnv
    from pixr.models import ModelArgs
    from pixr.task import get_task, get_task_cfgs
    torch.manual_seed(0)
    cfg = replace(get_task_cfgs()['read'](), model=ModelArgs(name=model_name, pretrained=False), amp=False)
    task = get_task('read')(cfg, device_env=DeviceEnv())
    torch.backends.cuda.matmul.allow_tf32 = False
    task.to(task.device)
    task.get_trainable_module().eval()  # no dropout, deterministic comparison
    return task


def _batch(task, B=2, N=64, T=24):
    ps = task.input_info['patch_size']
    ps = (ps, ps) if isinstance(ps, int) else ps
    patch_dim = ps[0] * ps[1] * (1 if task.input_info['image_fmt'] == 'L' else 3)
    g = torch.Generator().manual_seed(0)
    coord = torch.stack(torch.meshgrid(torch.arange(8), torch.arange(8), indexing='ij'), -1).reshape(-1, 2)
    text = torch.randint(5, 5000, (B, T), generator=g)
    batch = {
        'image_input': {
            'patches': torch.randn(B, N, patch_dim, generator=g),
            'patch_coord': coord[None].expand(B, -1, -1).contiguous(),
            'patch_valid': torch.arange(N)[None] < torch.tensor([[N], [N - 9]]),
        },
        'text_input': text,
        'text_target': text.clone(),
    }
    return task.prepare_batch(batch)


@pytest.mark.parametrize('model_name', ['cruller_naflex_siglip2_base', 'cronut_s16_tiktoken'])
@pytest.mark.parametrize('grad_ckpt', [False, True])
def test_fsdp_matches_plain(process_group, model_name, grad_ckpt):
    task = _make_task(model_name)
    ref_model = copy.deepcopy(task.get_trainable_module())
    batch = _batch(task)

    # reference: unwrapped model
    plain = copy.copy(task)
    object.__setattr__(plain, '_modules', dict(task._modules, trainable_module=ref_model))
    _, ref_losses = plain.training_forward(batch)
    ref_losses['loss'].backward()
    ref_grads = {n: p.grad.clone() for n, p in ref_model.named_parameters() if p.grad is not None}

    task.setup_ema(0.5)
    task.prepare_fsdp(grad_checkpointing=grad_ckpt)
    assert task.fsdp_enabled
    _, losses = task(batch)
    torch.testing.assert_close(losses['loss'], ref_losses['loss'], rtol=1e-5, atol=1e-5)
    losses['loss'].backward()
    model = task.get_trainable_module()
    for n, p in model.named_parameters():
        torch.testing.assert_close(p.grad.full_tensor(), ref_grads[n], rtol=1e-3, atol=1e-5, msg=n)

    # gathered full state dict keys match the plain model's
    state = task.model_state_dict()
    assert state['model'].keys() == ref_model.state_dict().keys()
    assert state['model_ema'].keys() == ref_model.state_dict().keys()

    # generation w/ gathered params matches the plain model
    start = [[task.tokenizer.convert_tokens_to_ids(task.task_start_token)]] * 2
    with task.full_params():
        out = task.generate(batch['image_input'], start, [task.tokenizer.eos_token_id], 6)
    ref_out = plain.generate(batch['image_input'], start, [task.tokenizer.eos_token_id], 6)
    assert out == ref_out

    # params are sharded again after full_params, training forward still works
    assert all(hasattr(p, 'full_tensor') for p in model.parameters())
    task(batch)[1]['loss'].backward()


@pytest.mark.parametrize('model_name', ['cruller_naflex_siglip2_base', 'cronut_s16_tiktoken'])
def test_fsdp_mixed_precision_generation_cache(process_group, model_name):
    """ W/ a bf16 MixedPrecisionPolicy FSDP2 casts forward inputs by rebuilding containers, KV caches updated in
    place inside a layer must survive (prefill fills every layer's cache, step decodes against it). """
    from torch.distributed.fsdp import MixedPrecisionPolicy
    task = _make_task(model_name)
    batch = _batch(task)
    task.prepare_fsdp(mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))
    model = task.get_trainable_module()
    start = [[task.tokenizer.convert_tokens_to_ids(task.task_start_token)]] * 2
    with torch.no_grad(), task.full_params():
        ids = batch['text_input'][:, :3]
        memory = model.encode_image(batch['image_input'])
        logits, state = model.prefill(memory, ids, torch.ones_like(ids, dtype=torch.bool))
        caches = tuple(state.cache)
        prefix_len = batch['image_input']['patches'].shape[1] if model.cfg.type == 'cronut' else 0
        for step in range(4):
            for original, cache in zip(caches, state.cache):
                assert cache is original
                assert cache.k is not None and cache.v is not None
                assert cache.k.shape[2] == cache.v.shape[2] == prefix_len + ids.shape[1]
            expected = model(batch['image_input'], ids)[:, -1]
            torch.testing.assert_close(logits, expected, atol=3e-2, rtol=1e-2)
            if step < 3:
                token = batch['text_input'][:, 3 + step:4 + step]
                ids = torch.cat([ids, token], dim=1)
                logits, state = model.step(token, state)
        out = task.generate(batch['image_input'], start, [], 6)
        # beam search: reorders the per layer caches every step, one beam must reproduce greedy
        assert task.generate(batch['image_input'], start, [], 6, num_beams=1) == out
        beams = task.generate(batch['image_input'], start, [], 6, num_beams=3)
    assert [len(o) for o in out] == [6, 6]  # no stop tokens, every step decoded
    assert [len(o) for o in beams] == [6, 6]
    # sharded params: every prefill / step would be a collective, generation must be refused w/ a clear error
    with pytest.raises(RuntimeError, match='full_params'):
        task.generate(batch['image_input'], start, [], 6)
    # nested scopes: the inner exit must not reshard under the outer scope
    with torch.no_grad(), task.full_params():
        with task.full_params():
            assert task.has_full_params(model)
        assert task.has_full_params(model)
        assert not any(hasattr(p, 'full_tensor') for p in model.parameters())  # still unsharded
        assert task.generate(batch['image_input'], start, [], 6) == out
    assert not task.has_full_params(model)
    assert all(hasattr(p, 'full_tensor') for p in model.parameters())  # resharded by the outermost exit
