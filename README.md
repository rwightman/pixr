# Pixel Reading (`pixr`)

Continuation of the original `pixparse` project (huggingface/pixparse, 2023), renamed and revived. Data
pipelines are provided by [`chugr`](https://github.com/rwightman/chugr) (formerly `chug`).

## Introduction

An open reproduction of OCR-free end-to-end document understanding models with open data.

Broadly focused on these model types:
* image encoder + text decoder w/ pixels and text tokens as input (as per Donut)
* image encoder + text decoder w/ only pixels as input (as per Pix2Struct)
* image encoder + text encoder-decoder w/ pixels and text tokens as input (as per PaLI/PaLI-X)

The training objectives and pretraining datasets will also be inspired by the associated papers above, but will mix and match. For example, we may train a Donut or PaLI-X style model with a Pix2Struct objective (masked document images w/ simplified HTML target text).

## Status (revive branch)

The `revive` branch ports `pixr` to `chugr` 0.2, NaFlexVit (native aspect / variable resolution) image encoders,
a self-contained text decoder, and fixes a number of issues that hampered the original DocVQA results:

* DocVQA fine-tune loss was applied to question + answer tokens (prompt masking used the task start token)
* train / eval image preprocessing mismatched (aspect preserving resize + pad vs stretch, long axis rotation)
* DocVQA eval decoded by re-tokenizing the generated string each step (BPE round trip is not idempotent)
* `scale_embedding` was forced on for BART decoders (copied from Donut's mBART, BART is trained w/o)
* `ModelArgs` command line overrides (image size, etc) were silently dropped
* pretrained BART cross-attention weights significantly slow learning w/ an image encoder, cross-attention is now
  re-initialized w/ zero init output projections by default (decoder starts as the pretrained LM)

## Models

Three model families share the same tasks, data pipelines, loss masking and (greedy / beam search, KV cached) generation:

* **Cruller** -- image encoder + autoregressive text decoder w/ cross-attention (Donut style)
* **Fritter** -- image encoder -> (optional m x m token merge) -> projector -> image token prefix for a decoder-only
  LM w/ prefix-LM attention (PaLI / LLaVA style)
* **Cronut** -- native decoder-only transformer over image patches and text (GenLIP style single trunk w/ 2D/1D
  interleaved MRoPE, qk-norm, gated attention, SwiGLU), trained from scratch w/ a tiktoken tokenizer

Model configs live in `src/pixr/models/configs`:

| config | image encoder | text decoder |
|---|---|---|
| `cruller_naflex_clip_base` | NaFlexVit B/16, CLIP style (pre-norm, cls token), OpenAI CLIP init, 4096 patches | BART base (6L) |
| `cruller_naflex_clip_base_8k` | as above, 8192 patches | BART base (6L) |
| `cruller_naflex_clip_base_ettin150m` | as above, 4096 patches | Ettin (ModernBERT) decoder 150M + cross-attn |
| `cruller_naflex_siglip2_base` | NaFlexVit B/16 SigLIP2 | BART base (6L) |
| `cruller_naflex_clip_large` | NaFlexVit L/14 CLIP style, OpenAI CLIP init, 6144 patches | BART large (8L) |
| `cruller_naflex_clip_large_ettin400m` | as above | Ettin decoder 400M + cross-attn |
| `cruller_naflex_siglip2_base_ettin150m` | NaFlexVit B/16 SigLIP2 | Ettin decoder 150M + cross-attn |
| `cruller_naflex_siglip2_so400m` | NaFlexVit SO400M/16 SigLIP2 | BART large (8L) |
| `fritter_naflex_siglip2_base_ettin150m` | NaFlexVit B/16 SigLIP2, 2x2 token merge | Ettin decoder 150M (prefix-LM) |
| `fritter_naflex_siglip2_base_ettin400m` | as above | Ettin decoder 400M (prefix-LM) |
| `fritter_naflex_siglip2_so400m_ettin1b` | NaFlexVit SO400M/16 SigLIP2, 2x2 token merge | Ettin decoder 1B (prefix-LM) |
| `cruller_naflex_siglip2_base_qwen3_0.6b` | NaFlexVit B/16 SigLIP2 | Qwen3 0.6B + zero init cross-attn every N layers |
| `fritter_naflex_siglip2_base_qwen3_0.6b` | NaFlexVit B/16 SigLIP2, 2x2 token merge | Qwen3 0.6B (prefix-LM) |
| `cronut_s16_tiktoken` / `cronut_b16_tiktoken` | -- (patch embed in trunk) | -- (single trunk), `cl100k_base` |
| `cruller_vit_clip_base_1280` | fixed size ViT B/16 CLIP @ 1280x960 | BART base (6L) |
| `cruller_hybrid_r50_base_1280` | fixed size hybrid R50 + ViT B/16 @ 1280x960 | BART base (6L) |

NaFlex models consume pages at their native aspect ratio, resized to fit a patch budget (`--task.model.max-seq-len`),
the budget can be changed between pretraining and fine-tuning.

Ettin (ModernBERT) decoders use 64 token sliding window attention in 2 of every 3 layers, a poor fit for a long
image prefix (Fritter), Qwen3 (full attention) is the recommended prefix-LM decoder.

### Attention backends

Padded NaFlex sequences can use a dense mask or a variable length kernel, selected per model config field:

| field | options |
|---|---|
| `image_encoder.attn_backend` | `sdpa` (dense mask), `varlen` (packed, CUDA), `flex` (FlexAttention block mask), batches w/o padding skip masking for all -- requires the timm `naflex-attn-backends` branch |
| `text_decoder.cross_attn_backend` | `sdpa`, `varlen`, `flex` |
| `text_decoder.self_attn_backend` | `sdpa` (dense prefix-LM mask), `flex` (prefix-LM block mask), Ettin / Qwen3 |
| `cronut.attn_backend` | `sdpa`, `flex` |

Override from the command line w/ `--task.model.image-attn-backend`, `--task.model.cross-attn-backend`, and
`--task.model.self-attn-backend` (decoder self-attention, or the Cronut trunk), e.g. `--task.model.image-attn-backend flex`.

### Tokenizers

HF tokenizers by name (`facebook/bart-base`, `Qwen/Qwen3-0.6B`, ...) or tiktoken encodings via `tiktoken:<encoding>`
(`tiktoken:cl100k_base`, `tiktoken:o200k_base`, ...), w/ eos / pad / bos and task tokens appended above the base vocab.

## Usage

Scripts in `scripts/` cover the main workflow, see each script for env var options.

```bash
# 1. pretrain (read all text on a page) on IDL, streamed from the HF hub
NUM_GPUS=2 MODEL=cruller_naflex_clip_base ./scripts/pretrain_idl.sh

# held out reading CER / WER
CHECKPOINT=output/pretrain_cruller_naflex_clip_base_idl/checkpoints/checkpoint-latest.pt ./scripts/eval_ocr.sh

# 2. fine-tune on DocVQA (ANLS on validation each eval interval)
CHECKPOINT=output/pretrain_cruller_naflex_clip_base_idl/checkpoints/checkpoint-latest.pt ./scripts/finetune_docvqa.sh

# 3. evaluate (validation ANLS, or SPLIT=test for an eval server submission file)
CHECKPOINT=output/finetune_docvqa/checkpoints/checkpoint-latest.pt ./scripts/eval_docvqa.sh
```

### Command line / config

Apps are configured by dataclasses parsed w/ [jsonargparse](https://github.com/omni-us/jsonargparse):

* `--task read|docvqa` selects the task (and the type of the `task` config section), default `read` for train
* nested fields map to dotted args, written w/ dashes: `--trainer.opt.learning-rate 1e-4` (underscores also accepted)
  * `--task.*` -- task: model, tokenizer, image transforms, dtype / amp, max text length, eval generation
  * `--trainer.*` -- optimization: intervals, warmup, optimizer / schedule (`--trainer.opt.*`), EMA, compile, grad ckpt,
    FSDP
  * `--train-data.*` / `--eval-data.*` -- chugr data args (source, format, split, batch size, workers, NaFlex batching)
* `--config file.yaml` loads a config (CLI args override), `--print-config` dumps the resolved config, `--help` lists all
* lists are passed as `--train-data.naflex-seq-lens [1024,2048,4096]`

The full resolved config is written to `<output_dir>/<experiment>/config.yaml` and can be passed back via `--config`.

### Distributed / compile

For short test / train / eval runs on two GPUs, use the scenario launcher from an environment with editable
`pixr`, `chugr`, and the local `timm` installed:

```bash
python scripts/run_review.py --list
python scripts/run_review.py all --validate                 # parse app args, no GPU jobs
python scripts/run_review.py smoke --dry-run                # inspect commands
python scripts/run_review.py smoke --devices 0,1
python scripts/run_review.py fsdp_compile naflex_repeat prefix_flex
python scripts/run_review.py data_ranks                     # check rank streams are distinct
python scripts/run_review.py tests parity
python scripts/run_review.py all
```

The launcher runs scenarios sequentially through `pixr.app.train`, `pixr.app.eval`, pytest and
`scripts/check_distributed.py`. Scenarios cover DDP / FSDP2, gradient accumulation, compile, checkpointing, EMA,
NaFlex batching, all three model families, attention backends, checkpoint resume and DocVQA fine-tuning.
`data_ranks` checks actual resampled WDS sample identities across ranks, without loading a model or decoding PDFs.
It fails if any two ranks receive an identical stream. App runs also fail if no updates or non-finite losses are logged.
`parity` compares FP32 DDP / FSDP2 model and EMA parameters and fails above a relative L2 difference of `1e-5`;
its resumed losses and generated outputs are also recorded for inspection.

Defaults are short execution checks: random initialization, 64 global samples per interval, batch 2 per GPU,
accumulation 2, 256 image patches and 128 text tokens. `--pretrained` enables hub weight initialization for the
app runs; tokenizers and Qwen3 / Ettin configs can still require downloads without it. `pretrained_tests` downloads
HF reference weights; `docvqa` uses the HF DocVQA dataset. `--samples`, `--batch-size`, `--accum`, `--patches`,
`--text-length`, and `--workers` adjust the workload. These short random-weight runs do not measure convergence.

The default data paths are sibling `../data/idl/idl-train-000{01..07}.tar` for training and shard `00000` for eval;
override with `--train-source` / `--eval-source`. `naflex_repeat` needs at least `GPUs * workers` training shards.
Logs, exact commands, resolved configs, checkpoints and `summary.json` go into a new `../runs/review-<timestamp>`
directory (`--output` overrides it). Each command has a 20-minute timeout (`--timeout` in seconds). Failed scenarios
are logged and the remaining scenarios continue; `--fail-fast` stops at the first failure. The launcher exits
nonzero on failure and prints a `tail -f` command for the active log.

Distributed runs (`torch.distributed.run`) use DDP by default, `--trainer.fsdp true` shards params, grads and
optimizer state w/ FSDP2 (`fully_shard` per encoder block / decoder layer + root, models list their blocks in
`fsdp_shard_modules()`). The handling follows the OpenCLIP `TrainingTask` (`prepare_distributed` / `prepare_fsdp` /
`compile` on the task):

* FSDP mixed precision is a `MixedPrecisionPolicy` (bf16 params, fp32 gradient reduction), autocast is suppressed
* `--trainer.grad-checkpointing` w/ FSDP applies composable activation checkpointing per block (inline otherwise)
* `--trainer.compile true --trainer.compile-strategy task|model|blocks`: `task` compiles the task training forward
  (model + loss), `model` the model before DDP, `blocks` the blocks in place before DDP (checkpoint recompute stays
  in eager autograd, bounding compiled backward memory). W/ FSDP, blocks are always compiled inside `prepare_fsdp()`,
  ordered compile -> composable checkpointing -> `fully_shard` (checkpoint hooks on the compiled wrapper so recompute
  re-enters the FSDP all-gather hooks), `--compile-mode` / `--compile-dynamic` / `--compile-backend` pass through
* eval and train metrics run w/ the params gathered (`task.full_params()`), so ranks decode independently
* checkpoints are gathered full state dicts (model, EMA, optimizer), loadable w/o FSDP, model / EMA weights are
  interchangeable between FSDP and DDP runs (optimizer state is not)

`scripts/check_distributed.py` checks single / DDP / FSDP parity (loss trajectories, final params), grad
checkpointing, accumulation, compile, EMA, resume and generation through the Trainer on synthetic batches.

### LR scaling

Params are grouped by component via the model's `param_layer_map()`: `encoder` (image encoder), `bridge`
(cross-attention / prefix projector), `decoder` (text decoder / LM trunk). Each has an LR multiplier
(`--trainer.opt.encoder-lr-scale`, `--trainer.opt.bridge-lr-scale`, `--trainer.opt.decoder-lr-scale`) and the
encoder / decoder support layer-wise LR decay (`--trainer.opt.encoder-layer-decay 0.75`,
`--trainer.opt.decoder-layer-decay`). `--trainer.opt.lr-scale-overrides '{"embed_tokens": 2.0}'` adds multipliers
by param name regex. Training pretrained encoders at full LR from the start wrecks their features before the decoder
learns to use them (reading never takes off), `scripts/pretrain_idl.sh` uses `ENC_LR_SCALE=0.1`.

### LM loss

The LM head + cross-entropy is fused and chunked over tokens by default (`--task.loss-chunk-size 4096`, 0 to
materialize full logits), padding is dropped before the head GEMM and each chunk's logits are recomputed in backward,
so the [tokens, vocab] logits (~150k vocab for Qwen3) are never materialized. `--task.z-loss-weight` (e.g. `1e-4`) adds
a z-loss (mean squared logsumexp), `--task.label-smoothing` is supported by both paths.

### NaFlex batching

For NaFlex models the train loader can draw a sequence length per batch and size the batch to a token budget, w/
samples resized to the batch's patch budget (rank-identical schedule, from OpenCLIP's streaming NaFlex loader):

```bash
--train-data.naflex-seq-lens [1024,2048,4096] --train-data.naflex-seq-len-probs [0.25,0.25,0.5] \
--train-data.naflex-max-tokens 65536 --train-data.naflex-text-tokens 512 --trainer.opt.loss-batch-scale sqrt
```

`--train-data.length-bucketing true` groups samples of similar target text length. Neither requires resampled shards.

Checkpoints include the resolved model config and the tokens added to the tokenizer, so fine-tuning and evaluation
only need `--checkpoint-path`, task tokens are appended and their embeddings initialized from the mean embedding.

## Code Organization

Within `src/pixr`:
* `app/` - CLI applications for training and evaluation
  * `app/train.py` - main training CLI entrypoint, will attempt to keep useable across tasks
  * `app/eval.py` - main evaluation CLI entrypoint
* `data/` - data loaders, image and text preprocessing
* `framework/` - lightweight train & evaluation scaffolding on top of canonical PyTorch
  * `framework/task.py` - `TrainingTask` base (model + objective + eval)
  * `framework/trainer.py` - `Trainer` (optimizer, LR schedule, AMP, grad accumulation, DDP / FSDP2 / compile setup,
    EMA, train state)
* `models/` - modelling code with associated factory methods and helpers
* `task/` - tasks (`read` pretraining, `docvqa` fine-tune), registered in `task/task_registry.py`
* `tokenizers/` - tokenizer creation and task token handling
* `generation.py` - batched, KV cached greedy and beam search decoding (`--task.eval-num-beams`)
* `utils/` - misc utils that don't have a home (`utils/cli.py` -- jsonargparse config / CLI parsing)

## Concepts & Terminology

Some terms and concepts used in this project that may be a bit unfamiliar.

### Task
A key organization concept in this project. Package the model with its loss, pre/post-processing, and optimization setup together for a given objective.

Examples of tasks conceptually:
  * Pretraining a Donut style (image enc + text dec) model on supervised (OCR annotation) doc-text dataset
  * Pretraining a Pix2Struct style (image enc + text dec) model w/ a dataset of webpage/screenshots and structured, simplified HTML
  * Pretraining a PaLI style (image enc + text enc-dec) model w/ prefix & masked-token completion on datasets as above
  * Fine-tuning any of the above pretrained models on a possibly large variety of downstream tasks
    * Semi-structured doc parsing - receipt, invoice, business cards, etc.
    * VQA
    * Captioning
    * ... and more

With the Task concept, the data pipeline exists outside the task. The data pipeline is coupled to the task by passing the pre-processing functions created within the task to the data pipeline on creation (`task.train_data_cfg()` / `task.eval_data_cfg()`).

A task is a `TrainingTask` (an `nn.Module` wrapping the model) implementing `training_forward(batch) -> (outputs, losses)`,
optional `train_metrics()`, and `evaluate(loader)`. The generic `Trainer` owns everything else (optimizer, schedule,
AMP, accumulation, setup ordering for DDP / FSDP2 / compile, EMA, checkpointing state). Tasks are agnostic to the model family,
Cruller, Fritter and Cronut share the same generation interface: `memory = model.encode_image(image_input)` (per image,
computed once), `logits, state = model.prefill(memory, prompt_ids, prompt_mask)`, `logits, state = model.step(token_ids, state)`
and `state.reorder(parent_idx)` to select / duplicate hypotheses (beam search). The per image memory and the per hypothesis
sequence state are separate (`pixr/models/kv_cache.py`): Cruller's cross-attention K / V are projected once inside each
decoder layer and shared by all hypotheses of an image (several prompts or beams per image, image-major order), only the
self-attention caches are reordered. Under FSDP, generation must run inside `task.full_params()`.

### Interval

You'll see the term 'interval' in the code, sometimes next to epoch. It's related, but an epoch means 'one complete pass of the dataset', an interval may be an epoch, but it may not. Interval is a span of training between checkpoints, ideally meaningful enough in duration to warrant evaluating and archiving each interval checkpoint.

In OpenCLIP development the term arose when using shard sampling with replacement, were the intervals between checkpoints were determined by limitations on job durations or likelihood of crashes.
