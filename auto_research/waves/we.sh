#!/bin/bash
# wave E：SWA1 下模型走 KV relay 的捷径、不学记忆（c1 的 delta rule 都卡在 relay 水平）
#   -> 上下文 dropout（每步 50% 概率整段不看过去）/ 从已经学会记忆的 c0 模型热启动再加注意力。代码冻结在 snapshots/we
cd /project/sds-rise/ethan/world_model/auto_research
S="--export=ALL,CODE=$PWD/snapshots/we"
C="--curriculum 1 --eval-splits curr3 test/n05 test/n10 test/n20 --ttt-layers all --ttt-loss mse --ttt-params down --ttt-fast-frac 0.25 --ttt-heads 8 --max-ctx 1"
W="--init /project/sds-rise/ethan/Finetune/world/auto_research/wd_mse_down_c0_f025_h8/ckpt_006000.pt --steps 3000"
sbatch -J we_c1_f025h8_cd05 $S train.slurm --run we_c1_f025h8_cd05 --ctx-drop 0.5 $C
sbatch -J we_warm_c1 $S train.slurm --run we_warm_c1 $W $C
sbatch -J we_warm_c1_cd05 $S train.slurm --run we_warm_c1_cd05 --ctx-drop 0.5 $W $C
