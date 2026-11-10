#!/usr/bin/env python3
"""Launch short, sequential test / train / eval scenarios on two GPUs.

Run with the Python environment containing editable pixr, chugr and timm installs:
    python scripts/run_review.py --list
    python scripts/run_review.py smoke --dry-run
    python scripts/run_review.py smoke
    python scripts/run_review.py fsdp_compile naflex_repeat prefix_flex

These are execution checks, not convergence benchmarks. Training defaults to random
weights, 256 patches, 128 text tokens and 64 samples / interval. Tokenizers and the
Qwen3 / Ettin architecture configs may still be downloaded. --pretrained loads weights.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SCENARIOS = {
    'data_ranks': 'check that resampled WDS streams differ across ranks (no model / image decoding)',
    'tests': 'pixr and chugr tests, plus the timm NaFlex backend / loading tests',
    'pretrained_tests': 'BART / Ettin / Qwen3 numerical comparisons with HF models (downloads weights)',
    'ddp': 'DDP, bf16 autocast, accumulated gradients, chunked loss, train metrics and eval',
    'ddp_compile': 'DDP with block compile and inline activation checkpointing',
    'ddp_task_compile': 'DDP with task compile and activation checkpointing',
    'fsdp': 'FSDP2, bf16 policy, accumulated gradients, composable checkpointing and EMA eval',
    'fsdp_no_reshard': 'FSDP2 keeping block parameters gathered between forward and backward',
    'fsdp_compile': 'FSDP2 with block compile, composable checkpointing and EMA eval',
    'naflex': 'DDP with varying patch budgets / batch sizes, length bucketing and loss scaling',
    'naflex_repeat': 'FSDP2 with NaFlex batching and repeating shards (no replacement)',
    'cruller_ettin': 'cross-attention Cruller / Ettin under FSDP2',
    'cruller_qwen3': 'cross-attention Cruller / Qwen3 under FSDP2',
    'fritter_ettin': 'prefix-LM Fritter / Ettin with token merge under FSDP2',
    'fritter_qwen3': 'prefix-LM Fritter / Qwen3 with token merge under FSDP2',
    'cronut': 'native decoder-only Cronut under FSDP2',
    'varlen': 'packed NaFlex encoder and cross-attention under FSDP2',
    'flex': 'FlexAttention encoder and cross-attention under FSDP2',
    'prefix_flex': 'Fritter / Qwen3 with FlexAttention prefix-LM under DDP autocast',
    'cronut_flex': 'Cronut with FlexAttention under FSDP2',
    'resume': 'FSDP2 + EMA: train, resume optimizer state, standalone model and EMA eval',
    'docvqa': 'read checkpoint -> DocVQA fine-tuning and distributed eval (downloads HF dataset)',
    'parity': 'deterministic FP32 DDP / FSDP2 comparison, resume and unequal-rank generation',
}
SUITES = {
    'smoke': ['data_ranks', 'ddp', 'fsdp', 'fritter_qwen3', 'cronut', 'resume'],
    'training': [s for s in SCENARIOS if s not in ('data_ranks', 'tests', 'pretrained_tests', 'docvqa', 'parity')],
    'all': list(SCENARIOS),
}


def options(**kwargs):
    args = []
    for key, value in kwargs.items():
        if isinstance(value, (bool, list)):
            value = json.dumps(value)
        args.extend(['--' + key, str(value)])
    return args


def build_scenarios(args, output):
    python = sys.executable
    launch = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(args.gpus)]
    checkpoints = output / 'experiments'
    base = {
        'task': 'read',
        'task.model.name': args.model,
        'task.model.pretrained': args.pretrained,
        'task.model.max-seq-len': args.patches,
        'task.model.text-max-length': args.text_length,
        'task.text-max-length': args.text_length,
        'task.dtype': 'bfloat16',
        'task.amp': True,
        'task.loss-chunk-size': 64,
        'task.num-metrics-samples': 2,
        'task.metrics-max-new-tokens': 8,
        'task.eval-max-new-tokens': 16,
        'train-data.source': args.train_source,
        'train-data.format': 'wds',
        'train-data.num-samples': args.samples,
        'train-data.batch-size': args.batch_size,
        'train-data.num-workers': args.workers,
        'train-data.resampled': True,
        'eval-data.source': args.eval_source,
        'eval-data.format': 'wds',
        'eval-data.num-samples': 8,
        'eval-data.batch-size': args.batch_size,
        # WDS eval num_samples is a per-worker limit; zero workers gives an exact limit.
        'eval-data.num-workers': 0,
        'trainer.num-intervals': 1,
        'trainer.num-warmup-intervals': 0,
        'trainer.opt.learning-rate': 1e-4,
        'trainer.opt.encoder-lr-scale': 0.1,
        'trainer.opt.grad-accum-steps': args.accum,
        'trainer.log-frequency': 1,
        'trainer.metrics-frequency': 4,
        'checkpoint-keep': 1,
        'output-dir': checkpoints,
    }
    fsdp = {'trainer.fsdp': True, 'trainer.grad-checkpointing': True}
    ema = {'trainer.ema-decay': 0.9, 'eval-ema': True}
    compile_blocks = {'trainer.compile': True, 'trainer.compile-strategy': 'blocks'}
    naflex = {
        'train-data.naflex-seq-lens': [args.patches // 2, args.patches, args.patches * 2],
        'train-data.naflex-max-tokens': args.batch_size * (args.patches * 2 + args.text_length),
        'train-data.naflex-text-tokens': args.text_length,
        'train-data.naflex-batch-divisor': 1,
        'train-data.length-bucketing': True,
        'trainer.opt.loss-batch-scale': 'linear',
    }

    def train(name, **overrides):
        return launch + ['-m', 'pixr.app.train'] + options(**{**base, 'experiment': name, **overrides})

    def evaluate(name, use_ema=False):
        return [python, '-m', 'pixr.app.eval'] + options(**{
            'task': 'read',
            'checkpoint-path': checkpoints / name / 'checkpoints' / 'checkpoint-latest.pt',
            'use-ema': use_ema,
            'task.eval-max-new-tokens': 16,
            'eval-data.source': args.eval_source,
            'eval-data.format': 'wds',
            'eval-data.num-samples': 8,
            'eval-data.batch-size': args.batch_size,
            'eval-data.num-workers': 0,
            'output-dir': output / name / ('eval_ema' if use_ema else 'eval_model'),
        })

    scenarios = {
        'data_ranks': [launch + [str(ROOT / 'scripts' / 'check_data_ranks.py'),
                                '--source', args.train_source, '--workers', str(args.workers)]],
        'ddp': [train('ddp')],
        'ddp_compile': [train('ddp_compile', **compile_blocks, **{'trainer.grad-checkpointing': True})],
        'ddp_task_compile': [train('ddp_task_compile', **{
            'trainer.compile': True, 'trainer.compile-strategy': 'task', 'trainer.grad-checkpointing': True})],
        'fsdp': [train('fsdp', **fsdp, **ema)],
        'fsdp_no_reshard': [train('fsdp_no_reshard', **fsdp, **{'trainer.fsdp-reshard-after-forward': False})],
        'fsdp_compile': [train('fsdp_compile', **fsdp, **ema, **compile_blocks)],
        'naflex': [train('naflex', **naflex)],
        'naflex_repeat': [train('naflex_repeat', **fsdp, **naflex, **{'train-data.resampled': False})],
    }
    models = {
        'cruller_ettin': 'cruller_naflex_siglip2_base_ettin150m',
        'cruller_qwen3': 'cruller_naflex_siglip2_base_qwen3_0.6b',
        'fritter_ettin': 'fritter_naflex_siglip2_base_ettin150m',
        'fritter_qwen3': 'fritter_naflex_siglip2_base_qwen3_0.6b',
        'cronut': 'cronut_s16_tiktoken',
    }
    for name, model in models.items():
        scenarios[name] = [train(name, **fsdp, **{'task.model.name': model})]
    for backend in ('varlen', 'flex'):
        scenarios[backend] = [train(backend, **fsdp, **{
            'task.model.image-attn-backend': backend, 'task.model.cross-attn-backend': backend})]
    scenarios['prefix_flex'] = [train('prefix_flex', **{
        'task.model.name': models['fritter_qwen3'], 'task.model.self-attn-backend': 'flex',
        'trainer.grad-checkpointing': True})]
    scenarios['cronut_flex'] = [train('cronut_flex', **fsdp, **{
        'task.model.name': models['cronut'], 'task.model.self-attn-backend': 'flex'})]
    scenarios['resume'] = [
        train('resume', **fsdp, **ema),
        launch + ['-m', 'pixr.app.train'] + options(**{
            'config': checkpoints / 'resume' / 'config.yaml',
            'resume': 'latest', 'trainer.num-intervals': 2}),
        evaluate('resume'), evaluate('resume', use_ema=True),
    ]
    # Task-specific fields must come from the DocVQA config, not ReadTaskCfg.
    docvqa = {k: v for k, v in base.items() if k not in (
        'task.num-metrics-samples', 'task.metrics-max-new-tokens')}
    docvqa.update({
        'task': 'docvqa', 'experiment': 'docvqa',
        'checkpoint-path': checkpoints / 'docvqa_read' / 'checkpoints' / 'checkpoint-latest.pt',
        'train-data.source': 'pixparse/docvqa-single-page-questions',
        'train-data.format': 'hfds', 'train-data.split': f'train[:{args.samples}]',
        'train-data.resampled': False,
        'eval-data.source': 'pixparse/docvqa-single-page-questions',
        'eval-data.format': 'hfds', 'eval-data.split': 'validation[:8]',
        'trainer.metrics-frequency': 0,
    })
    scenarios['docvqa'] = [train('docvqa_read'), launch + ['-m', 'pixr.app.train'] + options(**docvqa)]

    scenarios['tests'] = [
        [python, '-m', 'pytest', 'tests', '-q', '-k', 'not matches_hf and not reinit_cross_attn_starts_as_lm'],
        [python, '-m', 'pytest', str(WORKSPACE / 'chugr' / 'tests'), '-q'],
        [python, '-m', 'pytest', '-q'] + [str(WORKSPACE / 'timm' / 'tests' / path) for path in (
            'test_models.py::test_naflexvit_attn_backends_match_sdpa',
            'test_models.py::test_naflexvit_varlen_falls_back_to_dense_mask',
            'test_models.py::test_naflexvit_varlen_rejects_dropout',
            'test_checkpoint_loading.py::test_naflexvit_linear_patch_embed_adapts_in_chans',
            'test_checkpoint_loading.py::test_naflexvit_disabled_attn_pool_loads_strict',
        )],
    ]
    scenarios['pretrained_tests'] = [[
        python, '-m', 'pytest', 'tests/test_models.py', '-q',
        '-k', 'matches_hf or reinit_cross_attn_starts_as_lm',
    ]]
    parity_dir = output / 'parity'
    parity_args = [
        '--work-dir', str(parity_dir), '--model', args.model, '--fp32', '--steps', '6',
        '--global-batch', str(args.batch_size * args.gpus), '--patches', '256',
        '--text-len', str(args.text_length), '--accum', str(args.accum),
        '--ema', '--resume-check', '--eval-check', '--metrics-every', '3',
    ]
    parity_script = str(ROOT / 'scripts' / 'check_distributed.py')
    scenarios['parity'] = [
        launch + [parity_script, '--mode', mode, '--tag', mode] + parity_args for mode in ('ddp', 'fsdp')
    ] + [[python, parity_script, '--work-dir', str(parity_dir), '--compare', 'ddp', 'fsdp',
          '--max-param-rel-diff', '1e-5']]
    return scenarios


def validate_commands(scenarios):
    """Parse app arguments without constructing models, loaders, process groups or CUDA contexts."""
    from pixr.app.eval import EvalCfg
    from pixr.app.train import TrainCfg
    from pixr.task import get_task_cfgs
    from pixr.utils.cli import parse_cfg

    configs = {}
    count = 0
    for commands in scenarios.values():
        for command in commands:
            app = next((m for m in ('pixr.app.train', 'pixr.app.eval') if m in command), None)
            if app is None:
                continue
            argv = command[command.index(app) + 1:]
            if '--config' in argv:
                # Resume's config is written by its preceding train command at execution time.
                i = argv.index('--config')
                original = configs[argv[i + 1]]
                argv = original + argv[:i] + argv[i + 2:]
            cfg = parse_cfg(TrainCfg if app.endswith('train') else EvalCfg, argv=argv,
                            task_cfgs=get_task_cfgs(), default_task='read')
            if app.endswith('train'):
                configs[str(Path(cfg.output_dir) / cfg.experiment / 'config.yaml')] = argv
            count += 1
    # pytest commands, collect only, so renamed / removed test ids fail here and not after the GPU scenarios
    num_tests = 0
    for commands in scenarios.values():
        for command in commands:
            if command[1:3] != ['-m', 'pytest']:
                continue
            result = subprocess.run(command + ['--collect-only'], cwd=ROOT, capture_output=True, text=True)
            if result.returncode != 0:
                raise SystemExit(f'pytest collection failed for {shlex.join(command)}:\n{result.stdout[-2000:]}')
            num_tests += 1
    print(f'Validated {count} train / eval commands and collected {num_tests} pytest commands (no GPU jobs launched).')


def run_command(command, log_path, env, timeout):
    start = time.monotonic()
    with log_path.open('w') as log:
        log.write(shlex.join(command) + '\n\n')
        log.flush()
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
            status = 'ok' if code == 0 else 'failed'
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as ex:
            # Kill this job's whole process group, including torchrun workers / data loaders.
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            if isinstance(ex, KeyboardInterrupt):
                raise
            status, code = 'timeout', 124
    result = {'status': status, 'returncode': code, 'seconds': round(time.monotonic() - start, 2), 'log': str(log_path)}
    if code == 0 and 'pixr.app.train' in command:
        losses = [float(v) for v in re.findall(r'\bloss:\s*(\S+)', log_path.read_text(errors='replace'))]
        if not losses or not all(math.isfinite(v) for v in losses):
            result.update(status='failed', reason='no optimizer updates logged or non-finite training loss')
        else:
            result.update(first_loss=losses[0], last_loss=losses[-1], logged_updates=len(losses))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scenarios', nargs='*', help='scenario names, smoke (default), training, or all')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='print exact commands; write nothing')
    parser.add_argument('--validate', action='store_true', help='parse train / eval arguments; do not launch jobs')
    parser.add_argument('--devices', help='CUDA_VISIBLE_DEVICES, e.g. 0,1 (default: inherit)')
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--model', default='cruller_naflex_siglip2_base', help='base model for non-variant scenarios')
    parser.add_argument('--pretrained', action='store_true', help='initialize app training from hub weights')
    parser.add_argument('--train-source', default=str(WORKSPACE / 'data/idl/idl-train-000{01..07}.tar'))
    parser.add_argument('--eval-source', default=str(WORKSPACE / 'data/idl/idl-train-00000.tar'))
    parser.add_argument('--samples', type=int, default=64, help='global training samples per interval')
    parser.add_argument('--batch-size', type=int, default=2, help='per GPU (NaFlex uses a token budget)')
    parser.add_argument('--accum', type=int, default=2)
    parser.add_argument('--patches', type=int, default=256)
    parser.add_argument('--text-length', type=int, default=128)
    parser.add_argument('--workers', type=int, default=2, help='training workers per rank')
    parser.add_argument('--output', type=Path, help='new directory; defaults to runs/review-<timestamp>')
    parser.add_argument('--timeout', type=float, default=1200, help='seconds per command, including compile / downloads')
    parser.add_argument('--fail-fast', action='store_true', help='stop on first failure; otherwise continue scenarios')
    args = parser.parse_args()
    if args.list:
        for name, description in SCENARIOS.items():
            print(f'{name:20} {description}')
        for name, members in SUITES.items():
            print(f'{name:20} {", ".join(members)}')
        return 0
    if (args.gpus < 2 or min(args.samples, args.batch_size, args.accum) < 1 or args.patches < 2
            or args.text_length < 2 or args.workers < 0 or args.timeout <= 0):
        parser.error('need >= 2 GPUs, positive samples / batch size / accumulation / timeout, '
                     'patches and text length >= 2, and workers >= 0')
    if args.samples < args.batch_size * args.gpus * args.accum:
        parser.error('--samples must cover at least one accumulated optimizer update')
    selected = []
    for name in args.scenarios or ['smoke']:
        if name not in SCENARIOS and name not in SUITES:
            parser.error(f'unknown scenario {name!r}; use --list')
        selected.extend(SUITES.get(name, [name]))
    selected = list(dict.fromkeys(selected))
    output = (args.output or WORKSPACE / 'runs' / datetime.now().strftime('review-%Y%m%d-%H%M%S-%f')).resolve()
    all_scenarios = build_scenarios(args, output)
    scenarios = {name: all_scenarios[name] for name in selected}
    if args.validate:
        validate_commands(scenarios)
    if args.dry_run:
        print(f'Working directory: {ROOT}\nOutput: {output}')
        for name, commands in scenarios.items():
            print(f'\n{name}: {SCENARIOS[name]}')
            for command in commands:
                print(shlex.join(command))
    if args.validate or args.dry_run:
        return 0

    env = dict(os.environ)
    if args.devices is not None:
        env['CUDA_VISIBLE_DEVICES'] = args.devices
        os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    env['PYTHONUNBUFFERED'] = '1'
    # check_distributed.py's FP32 comparison requires TF32 off, including DeviceEnv's explicit enable.
    env['NVIDIA_TF32_OVERRIDE'] = '0'
    import torch
    if torch.cuda.device_count() < args.gpus:
        parser.error(f'need {args.gpus} visible CUDA GPUs; found {torch.cuda.device_count()}')
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'commands': scenarios, 'python': sys.executable, 'torch': torch.__version__, 'cwd': str(ROOT),
        'devices': [torch.cuda.get_device_name(i) for i in range(args.gpus)],
        'environment': {k: env[k] for k in (
            'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NVIDIA_TF32_OVERRIDE') if k in env},
    }
    (output / 'commands.json').write_text(json.dumps(manifest, indent=2) + '\n')
    results = []
    failed = False
    print(f'Logs / configs / checkpoints: {output}', flush=True)
    for name, commands in scenarios.items():
        dependency_failed = False
        for index, command in enumerate(commands, 1):
            label = f'{name}-{index}'
            log_path = output / f'{label}.log'
            # The test commands are independent; resume / docvqa / parity commands have dependencies.
            if dependency_failed and name != 'tests':
                result = {'status': 'skipped', 'reason': 'preceding command failed'}
            else:
                print(f'[{label}] {SCENARIOS[name]}\n  tail -f {shlex.quote(str(log_path))}', flush=True)
                try:
                    result = run_command(command, log_path, env, args.timeout)
                except KeyboardInterrupt:
                    result = {'status': 'interrupted', 'log': str(log_path)}
                    results.append({'scenario': name, 'step': index, **result})
                    (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
                    return 130
                dependency_failed |= result['status'] != 'ok'
                failed |= result['status'] != 'ok'
            results.append({'scenario': name, 'step': index, **result})
            (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
            print(f'[{label}] {result["status"]}', flush=True)
            if failed and args.fail_fast:
                return 1
    print(f'Results: {output / "summary.json"}', flush=True)
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
