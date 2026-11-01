#!/bin/bash
# 用法：bash run_toys.sh <snapshot 目录> <配置文件> <gpu> <日志>：按行顺序跑 toy.py
cd "$1"
while read -r line; do
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /project/sds-rise/ethan/miniforge3/envs/world/bin/python -u toy.py $line --steps 4000 --batch 128 --gpu "$3" 2>&1 | grep -v -i warn
done < "$2" >> "$4"
