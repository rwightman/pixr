#!/usr/bin/env bash
# Evaluate a DocVQA fine-tuned checkpoint. SPLIT=validation reports ANLS, SPLIT=test writes a submission json
# for the DocVQA eval server (rrc.cvc.uab.es).
set -euo pipefail
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT}
SPLIT=${SPLIT:-validation}
NUM_GPUS=${NUM_GPUS:-1}

python -m torch.distributed.run --nproc_per_node "${NUM_GPUS}" -m pixr.app.eval \
  --task cruller_eval_docvqa \
  --checkpoint-path "${CHECKPOINT}" \
  --eval-data.source pixparse/docvqa-single-page-questions \
  --eval-data.split "${SPLIT}" \
  --eval-data.format hfds \
  --eval-data.batch-size 32 \
  --eval-data.num-workers 8 \
  "$@"
