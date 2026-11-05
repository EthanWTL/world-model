#!/bin/bash
# 2026-11-03：Slurm 上的 base（22357148）排了 13 小时没排上 -> 本地训（用户：这个方向本地可以用两张卡；数据处理这种快任务
# 空闲时可以多用一张；finetune 只能两张）。代码用 snapshots/r1。
#   1) base：本地两张卡 DDP，全局 batch 16，20000 步
#   2) 训完自动：评估 + 13 个 run 交 Slurm；r1_v / r1_qkv 在第一张卡、r1_q / r1_ctrl 在第二张卡上依次本地跑
#   GPUS=2,3 setsid nohup bash waves/r1_local.sh > /project/sds-rise/ethan/Finetune/world/auto_research/visual/logs/r1_local.log 2>&1 &
cd /project/sds-rise/ethan/world_model/auto_research/visual/snapshots/r1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=$GPUS /project/sds-rise/ethan/miniforge3/envs/world/bin/torchrun --standalone --nproc_per_node 2 -- \
  train_base.py --run base --steps 20000 || exit 1
cd ../..
G1=${GPUS%,*}; G2=${GPUS#*,}
LOCAL="$G1:r1_v $G2:r1_q $G1:r1_qkv $G2:r1_ctrl" SKIP_BASE=1 bash waves/r1.sh | tee -a /project/sds-rise/ethan/Finetune/world/auto_research/visual/r1_jobs.txt
