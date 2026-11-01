#!/bin/bash
# 等之前的 toy 串跑完，再在两张卡上跑 toy5a / toy5b
R=/project/sds-rise/ethan/world_model/auto_research/snapshots
L=/project/sds-rise/ethan/Finetune/world/auto_research/logs
while pgrep -f "run_toys.sh .*toy2 " > /dev/null; do sleep 30; done
bash $R/run_toys.sh $R/toy5 toy_configs5a.txt 4 $L/toy5a_local.log &
while pgrep -f "after_toy3.sh" > /dev/null || pgrep -f "run_toys.sh .*toy4 " > /dev/null; do sleep 30; done
bash $R/run_toys.sh $R/toy5 toy_configs5b.txt 0 $L/toy5b_local.log
wait
