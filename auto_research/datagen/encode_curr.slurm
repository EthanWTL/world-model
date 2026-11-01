#!/bin/bash
# 课程数据编码成 latent（GPU，任意卡型）：sbatch encode_curr.slurm
#SBATCH --job-name=curr-encode
#SBATCH --account=sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/project/sds-rise/ethan/Finetune/world/auto_research/logs/curr_encode_%A_%a.log
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
echo "$(date '+%F %T') $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd /project/sds-rise/ethan/world_model/auto_research/datagen
/project/sds-rise/ethan/miniforge3/envs/world/bin/python -u encode_latents.py --root /project/sds-rise/ethan/Dataset/ShellGameCurr --splits train --num-shards 64 --claim
