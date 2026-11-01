#!/bin/bash
# eval.py 的 Slurm 版：一张卡，参数原样传给 eval.py（和 python eval.py ... 一样，只是前面换成 sbatch eval.slurm）
#   sbatch eval.slurm --ckpt /project/sds-rise/ethan/Finetune/world/shellgame/ema_050000.pt
#   sbatch eval.slurm --ckpt ... --splits test/n05 test/n30 --num 16
# V100 不支持 bf16 和 FlashAttention，用 constraint 排除掉；剩下的卡 world 环境都能跑
# 日志在 Finetune/world/logs/eval_<jobid>.log；结果和 mp4 在 <ckpt 所在目录>/eval/<ckpt 名>_<key>/
#SBATCH --job-name=world-eval
#SBATCH --account=sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000|b200|rtxpro6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/project/sds-rise/ethan/Finetune/world/logs/eval_%j.log

export PYTHONNOUSERSITE=1  # 不然 ~/.local/lib/python3.11 里的包会盖掉 world 环境里的
echo "$(date '+%F %T') $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd /project/sds-rise/ethan/world_model
/project/sds-rise/ethan/miniforge3/envs/world/bin/python -u eval.py "$@"
