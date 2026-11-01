#!/bin/bash
# Render the Shell Game dataset as a Slurm job array: task i renders episodes i, i+SHARDS, i+2*SHARDS, ... of all splits.
# CPU only (SwiftShader WebGL + VP9 encoding in headless Chromium). Re-submitting resumes: finished videos are skipped,
# and a subset of tasks can be re-run with e.g. `sbatch --array=3,7 shellgame/render.slurm` (SHARDS stays 10).
#   mkdir -p /project/sds-rise/ethan/Dataset/ShellGame/logs && sbatch shellgame/render.slurm
#SBATCH --job-name=shellgame-render
#SBATCH --account=sds-rise
#SBATCH --partition=standard
#SBATCH --array=0-9
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/project/sds-rise/ethan/Dataset/ShellGame/logs/render_%A_%a.log

SHARDS=10  # must cover --array above
OUT=/project/sds-rise/ethan/Dataset/ShellGame
CHROME=/usr/bin/chromium-browser

echo "$(date '+%F %T') $(hostname) shard $SLURM_ARRAY_TASK_ID/$SHARDS, $SLURM_CPUS_PER_TASK cpus"
[ -x "$CHROME" ] || { echo "no $CHROME on $(hostname)"; exit 1; }
cd /project/sds-rise/ethan/world_model
/home/jqf8qm/miniforge3/bin/python -u shellgame/generate.py --out "$OUT" \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$SHARDS" --workers $((SLURM_CPUS_PER_TASK / 2)) \
    --chrome "$CHROME" --chrome-args="--no-sandbox --use-angle=swiftshader --enable-unsafe-swiftshader"
