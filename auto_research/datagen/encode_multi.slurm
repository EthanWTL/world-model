#!/bin/bash
# 多球数据编码成 latent（GPU，任意卡型）：--follow 边渲染边编（每个 episode 一个锁），本地卡也可以同时跑同一行 python
#SBATCH --job-name=multi-encode
#SBATCH --account=sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/project/sds-rise/ethan/Finetune/world/auto_research/logs/multi_encode_%A_%a.log
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
echo "$(date '+%F %T') $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd /project/sds-rise/ethan/world_model/auto_research/datagen
/project/sds-rise/ethan/miniforge3/envs/world/bin/python -u encode_latents.py --root /project/sds-rise/ethan/Dataset/ShellGameMulti \
    --splits $(for k in 1 2 3 4 5; do echo k$k/train k$k/test/n05 k$k/test/n10 k$k/test/n20 k$k/test/n30; done) \
    --batch 16 --follow
