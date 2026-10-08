#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${WORK_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Port 29064 on GPU 0 (second instance)
CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port 29220 \
    -m siv_wam.server \
    --config-name robotwin_b2 \
    --port 29064 \
    --save_root ./eval_server_vis/ > eval_server_logs/server_extra_gpu0_port29064.log 2>&1 &
echo "Extra server 1: GPU=0 PORT=29064 PID=$!"

sleep 2

# Port 29065 on GPU 1 (second instance)
CUDA_VISIBLE_DEVICES=1 nohup python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port 29221 \
    -m siv_wam.server \
    --config-name robotwin_b2 \
    --port 29065 \
    --save_root ./eval_server_vis/ > eval_server_logs/server_extra_gpu1_port29065.log 2>&1 &
echo "Extra server 2: GPU=1 PORT=29065 PID=$!"
