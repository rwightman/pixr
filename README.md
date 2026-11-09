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

Three model families share the same tasks, data pipelines, loss masking and (greedy, KV cached) generation:

* **Cruller** -- image encoder + autoregressive text decoder w/ cross-attention (Donut style)
* **Fritter** -- image encoder -> (optional m x m token merge) -> projector -> image token prefix for a decoder-only
  LM w/ prefix-LM attention (PaLI / LLaVA style)
* **Cronut** -- native decoder-only transformer over image patches and text (placeholder, not implemented yet)

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
| `cruller_vit_clip_base_1280` | fixed size ViT B/16 CLIP @ 1280x960 | BART base (6L) |
| `cruller_hybrid_r50_base_1280` | fixed size hybrid R50 + ViT B/16 @ 1280x960 | BART base (6L) |

NaFlex models consume pages at their native aspect ratio, resized to fit a patch budget (`--task.model.max-seq-len`),
the budget can be changed between pretraining and fine-tuning.

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

Any config field can be set from the command line w/ dotted names (`--task.opt.learning-rate 1e-4`), or from a yaml
file via `--config`. The full resolved config is written to `<output_dir>/<experiment>/config.yaml`.

Checkpoints include the resolved model config and the tokens added to the tokenizer, so fine-tuning and evaluation
only need `--checkpoint-path`, task tokens are appended and their embeddings initialized from the mean embedding.

## Code Organization

Within `src/pixr`:
* `app/` - CLI applications for training and evaluation
  * `app/train.py` - main training CLI entrypoint, will attempt to keep useable across tasks
  * `app/eval.py` - main evaluation CLI entrypoint
* `data/` - data loaders, image and text preprocessing
* `framework/` - lightweight train & evaluation scaffolding on top of canonical PyTorch
* `models/` - modelling code with associated factory methods and helpers
* `task/` - task wrappers for various objectives (model + loss fn + pre/post-processing + optimization nuances)
* `tokenizers/` - tokenizer creation and task token handling
* `generation.py` - batched, KV cached greedy decoding
* `utils/` - misc utils that don't have a home

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

With the Task concept, the data pipeline exists outside the task. Samples and targets are fed into the task via the step functions. The data pipeline is coupled to the task by passing the pre-processing functions created within the task to the data pipeline on creation.

### Interval

You'll see the term 'interval' in the code, sometimes next to epoch. It's related, but an epoch means 'one complete pass of the dataset', an interval may be an epoch, but it may not. Interval is a span of training between checkpoints, ideally meaningful enough in duration to warrant evaluating and archiving each interval checkpoint.

In OpenCLIP development the term arose when using shard sampling with replacement, were the intervals between checkpoints were determined by limitations on job durations or likelihood of crashes.
