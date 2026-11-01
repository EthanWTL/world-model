#!/bin/bash
# 等 toy3 那一串跑完（GPU 0），接着跑多 seed
while pgrep -f "run_toys.sh .*toy3" > /dev/null; do sleep 30; done
bash /project/sds-rise/ethan/world_model/auto_research/snapshots/run_toys.sh /project/sds-rise/ethan/world_model/auto_research/snapshots/toy4 toy_configs_seeds.txt 0 /project/sds-rise/ethan/Finetune/world/auto_research/logs/toy4_local.log
