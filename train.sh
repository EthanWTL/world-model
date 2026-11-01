#!/bin/bash
# 本地训练：在 GPUS 指定的卡上用 torchrun 起 train.py（每张卡一个进程），前台运行，放在 tmux 里跑
# 全局 batch 固定 16，train.py 会平分到每张卡；同样的命令再跑一次，会从 <out>/last.pt 接着训
#   bash train.sh                    # 默认 GPU 0,1
#   GPUS=0,1 bash train.sh --out /project/sds-rise/ethan/Finetune/world/shellgame_swa1 --window 1
GPUS=${GPUS:-6,7}


cd /project/sds-rise/ethan/world_model
CUDA_VISIBLE_DEVICES=$GPUS PYTHONNOUSERSITE=1 /project/sds-rise/ethan/miniforge3/envs/world/bin/torchrun \
    --standalone --nproc_per_node "$(echo "$GPUS" | tr ',' '\n' | wc -l)" train.py "$@"
