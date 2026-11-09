#!/usr/bin/env bash
# Evaluate full page reading (CER / WER) of a pretrain checkpoint on held out IDL docs (first page).
# NOTE the default pretrain script trains on shards 0001+, shard 00000 is held out.
set -euo pipefail
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT}
NUM_SAMPLES=${NUM_SAMPLES:-500}
SOURCE=${SOURCE:-"pipe:curl -s -f -L https://huggingface.co/datasets/pixparse/idl-wds/resolve/main/idl-train-00000.tar"}

python -m pixparse.app.eval \
  --task cruller_eval_ocr \
  --checkpoint-path "${CHECKPOINT}" \
  --eval-data.source "${SOURCE}" \
  --eval-data.num-samples "${NUM_SAMPLES}" \
  --eval-data.batch-size 16 \
  --eval-data.num-workers 1 \
  "$@"
