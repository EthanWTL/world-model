#!/bin/bash
# toy.py 的 Slurm 数组：第 i 个任务跑 toy_configs.txt 的第 i 行（从 0 数）
#   sbatch --array=0-13 toy.slurm [toy.py 的其他公共参数]
#SBATCH --job-name=toy
#SBATCH --account=sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/project/sds-rise/ethan/Finetune/world/auto_research/logs/toy_%A_%a.log

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cd ${CODE:-/project/sds-rise/ethan/world_model/auto_research}
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" toy_configs.txt)
echo "$(date '+%F %T') $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) | $ARGS $@"
/project/sds-rise/ethan/miniforge3/envs/world/bin/python -u toy.py $ARGS "$@"
