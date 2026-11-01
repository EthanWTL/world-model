#!/bin/bash
# 用法：bash run_evals.sh <gpu> <ckpt>...：依次对每个 ckpt 跑完整评估（EMA，n05-n30，tf+free，64 个，50 步）
cd /project/sds-rise/ethan/world_model/auto_research/snapshots/ev1
G=$1; shift
for ck in "$@"; do
  out=$(dirname $ck)/eval/$(basename $ck .pt)_ema.json
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 /project/sds-rise/ethan/miniforge3/envs/world/bin/python -u evaluate.py \
    --ckpt $ck --key ema --splits test/n05 test/n10 test/n20 test/n30 --modes tf free --num 64 --steps 50 --gpu $G --out $out
done
