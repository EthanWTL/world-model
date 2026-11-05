#!/bin/bash
# 渲染多球数据（CPU）：sbatch render_multi.slurm（重跑会跳过已经渲染好的）
#SBATCH --job-name=multi-render
#SBATCH --account=sds-rise
#SBATCH --partition=standard
#SBATCH --array=0-19
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/project/sds-rise/ethan/Finetune/world/auto_research/logs/multi_render_%A_%a.log
echo "$(date '+%F %T') $(hostname) shard $SLURM_ARRAY_TASK_ID"
cd /project/sds-rise/ethan/world_model/auto_research/datagen
PYTHONDONTWRITEBYTECODE=1 /home/jqf8qm/miniforge3/bin/python -u gen_multi.py --shard "$SLURM_ARRAY_TASK_ID" --num-shards 20 \
    --workers $((SLURM_CPUS_PER_TASK / 2)) \
    --chrome /usr/bin/chromium-browser --chrome-args="--no-sandbox --use-angle=swiftshader --enable-unsafe-swiftshader"
