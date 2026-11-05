#!/bin/bash
# ttt_train.py 的 Slurm 版：一张卡，参数原样传给 ttt_train.py（和 python ttt_train.py ... 一样，只是前面换成 sbatch ttt.slurm）
#   sbatch -J ttt_q ttt.slurm --run q_delta
#   sbatch -J ttt_k ttt.slurm --run k_delta --mem-proj k
# 代码默认用 world_model/ 里现在的。排队期间可能会改代码，可以先拷一份，提交时加 --export=ALL,CODE=<那份的目录>
# V100 不支持 bf16，用 constraint 排除掉；一个 run 大约 21 GB 显存，剩下的卡都放得下
# 日志在 Finetune/world/ttt/logs/<job 名>_<jobid>.log；checkpoint、log.jsonl、eval/ 在 Finetune/world/ttt/<run>/
# 到时间被杀掉的话，同一条命令再交一次就从 last.pt 接着跑
#SBATCH --job-name=ttt
#SBATCH --account=sds-rise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000|b200|rtxpro6000
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/project/sds-rise/ethan/Finetune/world/ttt/logs/%x_%j.log

export PYTHONNOUSERSITE=1  # 不然 ~/.local/lib/python3.11 里的包会盖掉 world 环境里的
echo "$(date '+%F %T') $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) code=${CODE:-world_model}"
cd ${CODE:-/project/sds-rise/ethan/world_model}
/project/sds-rise/ethan/miniforge3/envs/world/bin/python -u ttt_train.py "$@"
