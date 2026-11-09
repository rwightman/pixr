#!/usr/bin/env bash
# Fine-tune a pretrained (read) checkpoint on DocVQA, evaluate ANLS on validation each interval.
#
# Usage: CHECKPOINT=output/<pretrain exp>/checkpoints/checkpoint-latest.pt NUM_GPUS=2 ./scripts/finetune_docvqa.sh
#
# The model architecture & tokens come from the checkpoint. The image patch budget can be raised for
# fine-tuning (NaFlex), e.g. MAX_SEQ_LEN=6144.
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-2}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to a pretrain checkpoint}
BATCH_SIZE=${BATCH_SIZE:-16}  # per GPU
ACCUM=${ACCUM:-2}
NUM_INTERVALS=${NUM_INTERVALS:-20}  # epochs over the 39k DocVQA train questions
LR=${LR:-5e-5}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
EXPERIMENT=${EXPERIMENT:-finetune_docvqa}

python -m torch.distributed.run --nproc_per_node "${NUM_GPUS}" -m pixparse.app.train \
  --task cruller_finetune_docvqa \
  --checkpoint-path "${CHECKPOINT}" \
  --task.model.max-seq-len "${MAX_SEQ_LEN}" \
  --train-data.source pixparse/docvqa-single-page-questions \
  --train-data.split train \
  --train-data.format hfds \
  --train-data.batch-size "${BATCH_SIZE}" \
  --train-data.num-workers 12 \
  --eval-data.source pixparse/docvqa-single-page-questions \
  --eval-data.split validation \
  --eval-data.format hfds \
  --eval-data.batch-size 32 \
  --eval-data.num-workers 8 \
  --task.num-intervals "${NUM_INTERVALS}" \
  --task.num-warmup-intervals 1 \
  --task.opt.learning-rate "${LR}" \
  --task.opt.grad-accum-steps "${ACCUM}" \
  --task.log-frequency 50 \
  --eval-frequency 2 \
  --experiment "${EXPERIMENT}" \
  --resume latest \
  --tensorboard true \
  "$@"
