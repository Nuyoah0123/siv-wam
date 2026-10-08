#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-SIV-WAM}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NNODES="${NNODES:-1}"
if [[ "$NNODES" != 1 ]]; then
  : "${MASTER_ADDR:?Set MASTER_ADDR on every node}"
  : "${NODE_RANK:?Set NODE_RANK on every node}"
fi
extra=()

exec "${PYTHON:-python}" -m torch.distributed.run \
  --nproc_per_node="${NGPU:-1}" --nnodes="$NNODES" \
  --node_rank="${NODE_RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29501}" \
  -m siv_wam.train --config-name "${CONFIG_NAME:-robotwin_train}" "${extra[@]}" "$@"
