#!/bin/bash
# wave F：最佳配方 = 先 c0 学会记忆，再热启动打开注意力（we_warm_c1：free 1.00/1.00/.94/.89）。代码冻结在 snapshots/wf
#   1) 热启动时用更大的窗口（3 / 7 = base 原本的窗口）  2) 整套配方换 seed  3) 热启动训更久  4) c0 阶段 mse_both+f0.25+h8
# 本地（GPU 0/4）：
#   python train.py --run wf_warm_c7 --init $BEST --steps 3000 --max-ctx 7 $T $C --gpu 0
#   python train.py --run wf_warm_c3 --init $BEST --steps 3000 --max-ctx 3 $T $C --gpu 4
cd /project/sds-rise/ethan/world_model/auto_research
S="--export=ALL,CODE=$PWD/snapshots/wf"
R=/project/sds-rise/ethan/Finetune/world/auto_research
BEST=$R/wd_mse_down_c0_f025_h8/ckpt_006000.pt
C="--curriculum 1 --eval-splits curr3 test/n05 test/n10 test/n20"
T="--ttt-layers all --ttt-loss mse --ttt-params down --ttt-fast-frac 0.25 --ttt-heads 8"
sbatch -J wf_warm_c1_long $S train.slurm --run wf_warm_c1_long --init $BEST --steps 6000 --max-ctx 1 $T $C
for s in 1 2; do
  J=$(sbatch --parsable -J wf_c0_s$s $S train.slurm --run wf_c0_s$s --seed $s --max-ctx 0 $T $C)
  sbatch --dependency=afterok:$J -J wf_warm_c1_s$s $S train.slurm --run wf_warm_c1_s$s --seed $s --init $R/wf_c0_s$s/ckpt_006000.pt --steps 3000 --max-ctx 1 $T $C
done
sbatch -J wf_c0_both_f025h8 $S train.slurm --run wf_c0_both_f025h8 --max-ctx 0 --ttt-layers all --ttt-loss mse --ttt-params both --ttt-fast-frac 0.25 --ttt-heads 8 $C
