#!/bin/bash
# render_multi.slurm 的 shard 1..19 拆成 4 份的小任务（8 核、1 小时，排队时好塞进碎片化的空闲 CPU）：task t -> shard 1 + t/4, sub t%4
#SBATCH --job-name=multi-render-s
#SBATCH --account=sds-rise
#SBATCH --partition=standard
#SBATCH --array=0-75
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/project/sds-rise/ethan/Finetune/world/auto_research/logs/multi_render_s_%A_%a.log
SH=$((1 + SLURM_ARRAY_TASK_ID / 4)); SUB=$((SLURM_ARRAY_TASK_ID % 4))
echo "$(date '+%F %T') $(hostname) shard $SH sub $SUB"
cd /project/sds-rise/ethan/world_model/auto_research/datagen
PYTHONDONTWRITEBYTECODE=1 /home/jqf8qm/miniforge3/bin/python -u gen_multi.py --shard $SH --num-shards 20 --sub $SUB --num-subs 4 \
    --workers 4 --chrome /usr/bin/chromium-browser --chrome-args="--no-sandbox --use-angle=swiftshader --enable-unsafe-swiftshader"
