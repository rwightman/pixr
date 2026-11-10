""" Single / DDP / FSDP2 parity checks through the pixr Trainer + ReadTask on deterministic synthetic batches.

Each run trains `--steps` steps on the same global batches (split across ranks), records the (rank mean) loss per
step and saves the final full model (+ EMA) state. W/ `--fp32` (and NVIDIA_TF32_OVERRIDE=0) DDP and FSDP runs match
to ~1e-7 (final param relative diff), single process runs differ by dropout RNG streams.

    export NVIDIA_TF32_OVERRIDE=0
    python -m torch.distributed.run --nproc_per_node 2 scripts/check_distributed.py --mode ddp --fp32 --tag ddp
    python -m torch.distributed.run --nproc_per_node 2 scripts/check_distributed.py --mode fsdp --fp32 --tag fsdp \\
        --grad-ckpt --compile --ema --resume-check --eval-check --metrics-every 3
    python scripts/check_distributed.py --compare ddp fsdp

Options cover grad checkpointing, accumulation, compile strategies, EMA, resume from a mid-run checkpoint (the
resumed losses should track the original second half), and generation w/ gathered params (ranks decode different
numbers of batches / tokens, which would hang if any collective ran inside).
"""
import argparse
import json
import os
import time
from dataclasses import replace

import torch
import torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='single', choices=('single', 'ddp', 'fsdp'))
ap.add_argument('--tag', default='')
ap.add_argument('--work-dir', default='./output/check_distributed')
ap.add_argument('--compare', nargs='+', default=None, help='compare saved runs by tag, first is the reference')
ap.add_argument('--model', default='cruller_naflex_siglip2_base')
ap.add_argument('--fp32', action='store_true', help='no AMP / FSDP mixed precision, for tight parity')
ap.add_argument('--grad-ckpt', action='store_true')
ap.add_argument('--compile', action='store_true')
ap.add_argument('--strategy', default='task')
ap.add_argument('--accum', type=int, default=1)
ap.add_argument('--ema', action='store_true')
ap.add_argument('--no-reshard', action='store_true')
ap.add_argument('--steps', type=int, default=12)
ap.add_argument('--global-batch', type=int, default=8)
ap.add_argument('--patches', type=int, default=256)
ap.add_argument('--text-len', type=int, default=64)
ap.add_argument('--dynamic', action='store_true', help='NaFlex like varying patch / text lengths per step')
ap.add_argument('--metrics-every', type=int, default=0)
ap.add_argument('--resume-check', action='store_true')
ap.add_argument('--eval-check', action='store_true')
args = ap.parse_args()


def compare(tags):
    runs = {t: json.load(open(os.path.join(args.work_dir, f'{t}.json'))) for t in tags}
    for t, r in runs.items():
        line = f'{t:>16} ' + ' '.join(f'{x:.5f}' for x in r['losses'])
        line += f" | {r['time_per_step'] * 1000:.0f} ms/step {r['peak_gb']:.2f} GB"
        if 'resume_losses' in r:
            line += ' | resumed ' + ' '.join(f'{x:.5f}' for x in r['resume_losses'])
        print(line)
    states = {t: torch.load(os.path.join(args.work_dir, f'{t}.pt'), map_location='cpu') for t in tags}
    ref = tags[0]
    for t in tags[1:]:
        for key in ('model', 'model_ema'):
            if key not in states[ref] or key not in states[t]:
                continue
            x, y = states[ref][key], states[t][key]
            assert x.keys() == y.keys(), f'state dict keys differ: {set(x) ^ set(y)}'
            num = sum((x[k].float() - y[k].float()).norm() ** 2 for k in x) ** 0.5
            den = sum(x[k].float().norm() ** 2 for k in x) ** 0.5
            print(f'{key} rel diff {ref} vs {t}: {num / den:.2e}')


if args.compare:
    compare(args.compare)
    raise SystemExit(0)

from pixr.framework import DeviceEnv, Trainer, TrainerCfg, random_seed
from pixr.framework.config import OptimizationCfg
from pixr.models import ModelArgs
from pixr.task import get_task, get_task_cfgs

device_env = DeviceEnv()
world, rank = device_env.world_size, device_env.global_rank
assert (args.mode == 'single') == (world == 1), 'use --mode single w/o torch.distributed.run, ddp / fsdp with'
random_seed(0, rank=0)  # same init on all ranks
tag = args.tag or args.mode
os.makedirs(args.work_dir, exist_ok=True)
mid_checkpoint = os.path.join(args.work_dir, f'{tag}-mid.pt')


def make_task():
    torch.manual_seed(0)
    task_cfg = get_task_cfgs()['read']()
    task_cfg = replace(
        task_cfg,
        model=ModelArgs(name=args.model, pretrained=False),
        amp=not args.fp32,
        num_metrics_samples=2,
        metrics_max_new_tokens=16,
    )
    return get_task('read')(task_cfg, device_env=device_env)


def make_trainer(task):
    cfg = TrainerCfg(
        num_intervals=1,
        num_warmup_intervals=0.,
        log_frequency=10 ** 9,
        metrics_frequency=args.metrics_every,
        opt=OptimizationCfg(learning_rate=3e-4, grad_accum_steps=args.accum, encoder_lr_scale=0.1),
        grad_checkpointing=args.grad_ckpt,
        compile=args.compile,
        compile_strategy=args.strategy,
        ema_decay=0.9 if args.ema else None,
        fsdp=args.mode == 'fsdp',
        fsdp_reshard_after_forward=not args.no_reshard,
    )
    return Trainer(task, cfg, device_env=device_env)


def make_batch(step, micro, task, local=True):
    """ Deterministic batch, the global batch split across ranks (or all of it for single). """
    info = task.input_info
    ps = info['patch_size']
    ps = (ps, ps) if isinstance(ps, int) else ps
    patch_dim = ps[0] * ps[1] * (1 if info['image_fmt'] == 'L' else 3)
    B, N, T = args.global_batch, args.patches, args.text_len
    if args.dynamic:
        N = (12 + 2 * (step % 3)) ** 2
        T = T - 16 * (step % 4)
    g = torch.Generator().manual_seed(1000 * (step % 4) + micro)
    side = int(N ** 0.5)
    coord = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing='ij'), -1).reshape(-1, 2)
    lens = N - torch.randint(0, N // 4, (B,), generator=g)
    text = torch.randint(5, 5000, (B, T), generator=g)
    batch = {
        'image_input': {
            'patches': torch.randn(B, N, patch_dim, generator=g),
            'patch_coord': coord[None].expand(B, -1, -1).contiguous(),
            'patch_valid': torch.arange(N)[None] < lens[:, None],
        },
        'text_input': text,
        'text_target': text.clone(),
    }
    if local and world > 1:
        sl = slice(rank * B // world, (rank + 1) * B // world)
        batch = {
            'image_input': {k: v[sl] for k, v in batch['image_input'].items()},
            'text_input': batch['text_input'][sl],
            'text_target': batch['text_target'][sl],
        }
    return batch


class Loader:
    def __init__(self, task, start, steps):
        self.task, self.start, self.steps = task, start, steps

    def __iter__(self):
        for step in range(self.start, self.start + self.steps):
            for micro in range(args.accum):
                yield make_batch(step, micro, self.task)


def rank_mean(x: float) -> float:
    t = torch.tensor([x], device=device_env.device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(t)
        t /= world
    return t.item()


def train(trainer, task, start, steps):
    losses = []
    after_step = trainer._after_step

    def record(batch, outputs, step_losses):
        losses.append(rank_mean(step_losses['loss'].item()))
        return after_step(batch, outputs, step_losses)
    trainer._after_step = record
    trainer.train_interval(Loader(task, start, steps))
    trainer._after_step = after_step
    return losses


result = dict(vars(args))
task = make_task()
trainer = make_trainer(task)
trainer.setup(num_batches_per_interval=args.steps * args.accum * 2)
if rank == 0:
    print(f'[{tag}] trainable_module {type(task.trainable_module).__name__}, fsdp={task.fsdp_enabled}', flush=True)

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.time()
half = args.steps // 2
losses = train(trainer, task, 0, half)
if args.resume_check:
    state = trainer.state_dict()  # collective w/ FSDP, populated on rank 0
    if rank == 0:
        torch.save(state, mid_checkpoint)
    del state
    if world > 1:
        dist.barrier()
losses += train(trainer, task, half, args.steps - half)
torch.cuda.synchronize()
result['time_per_step'] = (time.time() - t0) / args.steps
result['peak_gb'] = torch.cuda.max_memory_allocated() / 2 ** 30
result['losses'] = losses

if args.resume_check:
    task2 = make_task()
    trainer2 = make_trainer(task2)
    trainer2.setup(num_batches_per_interval=args.steps * args.accum * 2, resume=mid_checkpoint)
    result['resume_losses'] = train(trainer2, task2, half, args.steps - half)
    del task2, trainer2

state = trainer.state_dict()
if rank == 0:
    torch.save({k: v for k, v in state.items() if k in ('model', 'model_ema')}, os.path.join(args.work_dir, f'{tag}.pt'))
    if args.resume_check:
        os.remove(mid_checkpoint)
del state

if args.eval_check:
    generated = []
    with task.full_params(use_ema=args.ema):
        for i in range(2 if rank == 0 else 1):
            batch = make_batch(100 + i, 0, task, local=False)
            image_input = {k: v[:4] for k, v in batch['image_input'].items()}
            start_id = task.tokenizer.convert_tokens_to_ids(task.task_start_token)
            generated.append(task.generate(
                image_input, [[start_id]] * 4, [task.tokenizer.eos_token_id], 8 + 8 * rank + 4 * i, use_ema=args.ema))
    result['generated'] = generated
    result['losses_after_eval'] = train(trainer, task, 0, 2)  # training continues after gathered eval

if rank == 0:
    print(json.dumps({k: v for k, v in result.items() if k != 'generated'}), flush=True)
    with open(os.path.join(args.work_dir, f'{tag}.json'), 'w') as f:
        json.dump(result, f)
if world > 1:
    dist.barrier()
    dist.destroy_process_group()
