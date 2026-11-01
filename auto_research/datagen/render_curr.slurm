#!/bin/bash
# 渲染课程数据（CPU）：sbatch render_curr.slurm
#SBATCH --job-name=curr-render
#SBATCH --account=sds-rise
#SBATCH --partition=standard
#SBATCH --array=0-3
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/project/sds-rise/ethan/Finetune/world/auto_research/logs/curr_render_%A_%a.log
echo "$(date '+%F %T') $(hostname) shard $SLURM_ARRAY_TASK_ID"
cd /project/sds-rise/ethan/world_model/auto_research/datagen
/home/jqf8qm/miniforge3/bin/python -u gen_curr.py --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4 --workers $((SLURM_CPUS_PER_TASK / 2)) \
    --chrome /usr/bin/chromium-browser --chrome-args="--no-sandbox --use-angle=swiftshader --enable-unsafe-swiftshader"
