#!/usr/bin/env bash
set -euo pipefail
NODE=${1:-0}
NUM_SHARDS=${NUM_SHARDS:-24}
MODE=${MODE:---offline}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOTWIN="${SIV_WAM_ROBOTWIN_ROOT:?Set SIV_WAM_ROBOTWIN_ROOT}"
LEROBOT="${SIV_WAM_DATASET_PATH:?Set SIV_WAM_DATASET_PATH}"
EXPORT="${SIV_WAM_SIV_EXPORT_ROOT:-${PROJECT_ROOT}/outputs/contact_siv_dataset}"
LOGS="${SIV_WAM_LOG_ROOT:-${PROJECT_ROOT}/outputs/c2s_batch_logs}"
PYTHON="${PYTHON:-python}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-500}"
mkdir -p "$EXPORT" "$LOGS"
cd "$PROJECT_ROOT"
for ((i = 0; i < 8; i++)); do
  SHARD=$((NODE * 8 + i)); GPU=$i
  nohup "$PYTHON" data_process/scripts/batch_export.py $MODE \
    --robotwin_root "$ROBOTWIN" --lerobot_root "$LEROBOT" \
    --export_root "$EXPORT" --shard_index "$SHARD" --num_shards "$NUM_SHARDS" \
    --gpus "$GPU" --episodes_per_task "$EPISODES_PER_TASK" \
    --depth_tolerance 0.05 --save_contacts --log_dir "$LOGS" \
    > "$LOGS/shard${SHARD}.out" 2>&1 &
  echo "launched shard $SHARD on gpu $GPU"
done
