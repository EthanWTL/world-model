#!/bin/bash
R=/project/sds-rise/ethan/world_model/auto_research/snapshots
while pgrep -f "run_toys.sh .*toy5 " > /dev/null; do sleep 30; done
bash $R/run_toys.sh $R/toy6 toy_configs6.txt 4 /project/sds-rise/ethan/Finetune/world/auto_research/logs/toy6_local.log
