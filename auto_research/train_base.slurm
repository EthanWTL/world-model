#!/bin/bash
# train_base.py 的 Slurm 版（本地没有两张空卡时用）：一个节点两张卡，参数原样传给 train_base.py；重交同一行从 last.pt 接着跑
#   sbatch -J mb_base train_base.slurm --run mb_base
#SBATCH --account=sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100_80gb|b200|rtxpro6000
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/project/sds-rise/ethan/Finetune/world/auto_research/logs/%x_%j.log

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
echo "$(date '+%F %T') $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd ${CODE:-/project/sds-rise/ethan/world_model/auto_research}
/project/sds-rise/ethan/miniforge3/envs/world/bin/torchrun --standalone --nproc_per_node 2 train_base.py "$@"
