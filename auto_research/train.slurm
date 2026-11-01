#!/bin/bash
# train.py 的 Slurm 版：一张卡，参数原样传给 train.py
#   sbatch -J dot_both_all train.slurm --run dot_both_all --ttt-layers all
# 显存要 ≥80 GB（a100_80gb / b200 / rtxpro6000）；时间到了重新 sbatch 同一行会从 last.pt 接着跑
#SBATCH --account=sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb|b200|rtxpro6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/project/sds-rise/ethan/Finetune/world/auto_research/logs/%x_%j.log

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
echo "$(date '+%F %T') $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd ${CODE:-/project/sds-rise/ethan/world_model/auto_research}  # CODE=.../snapshots/wN：用那一波冻结的代码
/project/sds-rise/ethan/miniforge3/envs/world/bin/python -u train.py "$@"
