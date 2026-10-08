START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}

save_root='visualization/'
mkdir -p $save_root

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    -m siv_wam.server \
    --config-name robotwin \
    --port $START_PORT \
    --save_root $save_root


