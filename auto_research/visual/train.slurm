#!/bin/bash
# train.py 的 Slurm 版：一张卡，参数原样传给 train.py；时间到了重交同一行会从 last.pt 接着跑
#   sbatch -J v_h4 train.slurm --run v_h4 --mem-proj v
#   CODE=.../snapshots/r1 用那一波冻结的代码
#SBATCH --account=sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb|b200|rtxpro6000|a6000|a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/project/sds-rise/ethan/Finetune/world/auto_research/visual/logs/%x_%j.log
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
echo "$(date '+%F %T') $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd ${CODE:-/project/sds-rise/ethan/world_model/auto_research/visual}
/project/sds-rise/ethan/miniforge3/envs/world/bin/python -u train.py "$@"
