#!/bin/bash
# 渲染 + 编码旋转方块数据（GPU，任意卡型）：每个任务抢没人领的 shard（64 个），领完就退出；重交会跳过已经有的 latent
#   sbatch encode.slurm         （先 python build.py --meta）
#SBATCH --job-name=cube-encode
#SBATCH --account=sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-15
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/project/sds-rise/ethan/Finetune/world/auto_research/visual/logs/cube_encode_%A_%a.log
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
echo "$(date '+%F %T') $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd /project/sds-rise/ethan/world_model/auto_research/visual/datagen
/project/sds-rise/ethan/miniforge3/envs/world/bin/python -u build.py --encode --claim
