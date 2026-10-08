#!/bin/bash
# Launch 10 SIV-WAM server instances for RoboTwin evaluation
# Uses our B2 finetuned checkpoint (step 1200)
# Ports: 29056 - 29065 (10 servers)
# Each GPU runs 1-2 instances (8 GPUs total on this node)

set -e

START_PORT=${START_PORT:-29056}
NUM_SERVERS=${NUM_SERVERS:-10}
LOG_DIR="${LOG_DIR:-./eval_server_logs}"
MASTER_PORT_BASE=29200

mkdir -p $LOG_DIR

batch_time=$(date +%Y%m%d_%H%M%S)
echo "Launching $NUM_SERVERS server instances, ports $START_PORT-$((START_PORT + NUM_SERVERS - 1))"

pid_file="$LOG_DIR/server_pids_${batch_time}.txt"
> "$pid_file"

for i in $(seq 0 $((NUM_SERVERS - 1))); do
    # Assign GPU: first 8 use GPUs 0-7, extras wrap around
    gpu_id=$(( i % 8 ))
    port=$((START_PORT + i))
    master_port=$((MASTER_PORT_BASE + i))
    log_file="${LOG_DIR}/server_${i}_gpu${gpu_id}_port${port}_${batch_time}.log"

    echo "Server $i: GPU=$gpu_id PORT=$port MASTER_PORT=$master_port -> $log_file"

    CUDA_VISIBLE_DEVICES=$gpu_id \
    nohup python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port $master_port \
        -m siv_wam.server \
        --config-name robotwin_b2 \
        --port $port \
        --save_root ./eval_server_vis/ > "$log_file" 2>&1 &

    pid=$!
    echo "$pid" >> "$pid_file"
    echo "  -> PID: $pid"
    sleep 1
done

echo ""
echo "All $NUM_SERVERS server instances launched."
echo "PIDs saved to: $pid_file"
echo ""
echo "Wait ~60s for models to load, then run the evaluation clients."
echo "To stop all servers: kill \$(cat $pid_file)"
