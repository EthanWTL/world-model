#!/bin/bash
# evaluate.py 的 Slurm 版：一张卡，参数原样传给 evaluate.py
#   sbatch -J ev_base eval.slurm --ckpt .../visual/base/ema_020000.pt --max-ctx -1
#SBATCH --account=sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000|b200|rtxpro6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/project/sds-rise/ethan/Finetune/world/auto_research/visual/logs/%x_%j.log
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
echo "$(date '+%F %T') $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd ${CODE:-/project/sds-rise/ethan/world_model/auto_research/visual}
/project/sds-rise/ethan/miniforge3/envs/world/bin/python -u evaluate.py "$@"
