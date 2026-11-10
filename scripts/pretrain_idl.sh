#!/usr/bin/env bash
# Pretrain (read all page text) on IDL (+ optionally PDFA) webdataset shards.
#
# Usage: NUM_GPUS=2 ./scripts/pretrain_idl.sh [extra args...]
#
# Shards are streamed from the HF hub by default (IDL is ~2.6TB). Set IDL_SOURCE to local shards if available.
# Donut pretraining reference scale: ~200k steps @ batch 196 (~40M samples). Default below is 8M samples
# (40 intervals x 200k), roughly 4 days on 2x RTX PRO 6000 (Max-Q) w/ cruller_naflex_clip_base @ 4096 patches.
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-2}
MODEL=${MODEL:-cruller_naflex_siglip2_base}
BATCH_SIZE=${BATCH_SIZE:-16}  # per GPU
ACCUM=${ACCUM:-4}  # global batch = NUM_GPUS * BATCH_SIZE * ACCUM
NUM_INTERVALS=${NUM_INTERVALS:-40}
SAMPLES_PER_INTERVAL=${SAMPLES_PER_INTERVAL:-200000}
LR=${LR:-3e-4}
ENC_LR_SCALE=${ENC_LR_SCALE:-0.1}  # pretrained image encoder LR multiplier (1.0 wrecks pretrained features early)
SYNTH_PROB=${SYNTH_PROB:-0.0}  # synthetic text renderings of the page text (1.0 for a reading warmup)
WORKERS=${WORKERS:-24}
CURL="curl -s -f -L --http1.1 --retry 10 --retry-all-errors"
IDL_SOURCE=${IDL_SOURCE:-"pipe:${CURL} https://huggingface.co/datasets/pixparse/idl-wds/resolve/main/idl-train-0{0001..2999}.tar"}
EXPERIMENT=${EXPERIMENT:-pretrain_${MODEL}_idl}

python -m torch.distributed.run --nproc_per_node "${NUM_GPUS}" -m pixr.app.train \
  --task read \
  --task.model.name "${MODEL}" \
  --train-data.source "${IDL_SOURCE}" \
  --train-data.resampled true \
  --train-data.num-samples "${SAMPLES_PER_INTERVAL}" \
  --train-data.batch-size "${BATCH_SIZE}" \
  --train-data.num-workers "${WORKERS}" \
  --trainer.num-intervals "${NUM_INTERVALS}" \
  --trainer.num-warmup-intervals 1 \
  --trainer.opt.learning-rate "${LR}" \
  --trainer.opt.encoder-lr-scale "${ENC_LR_SCALE}" \
  --task.synth-prob "${SYNTH_PROB}" \
  --trainer.opt.grad-accum-steps "${ACCUM}" \
  --trainer.opt.weight-decay 0.05 \
  --trainer.log-frequency 50 \
  --trainer.metrics-frequency 1000 \
  --experiment "${EXPERIMENT}" \
  --resume latest \
  --tensorboard true \
  "$@"
