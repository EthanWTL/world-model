#!/bin/bash
# wave D：用户的"留一些 slot 兜底 + 剩下的分 head"（toy：mse_down f0.25 → N30 0.99，h4 → 0.95）。代码冻结在 snapshots/wd
cd /project/sds-rise/ethan/world_model/auto_research
S="--export=ALL,CODE=$PWD/snapshots/wd"
C="--curriculum 1 --eval-splits curr3 test/n05 test/n10 test/n20 --ttt-layers all --ttt-loss mse --ttt-params down"
sbatch -J wd_mse_down_c1_f025 $S train.slurm --run wd_mse_down_c1_f025 --max-ctx 1 --ttt-fast-frac 0.25 $C
sbatch -J wd_mse_down_c1_f025_h8 $S train.slurm --run wd_mse_down_c1_f025_h8 --max-ctx 1 --ttt-fast-frac 0.25 --ttt-heads 8 $C
sbatch -J wd_mse_down_c1_h8 $S train.slurm --run wd_mse_down_c1_h8 --max-ctx 1 --ttt-heads 8 $C
sbatch -J wd_mse_down_c0_f025_h8 $S train.slurm --run wd_mse_down_c0_f025_h8 --max-ctx 0 --ttt-fast-frac 0.25 --ttt-heads 8 $C
# toy：TTT 放前面的层好、放最后的层完全学不会 -> 世界模型验证（前 4 层 vs 后 4 层，都 f0.25）
sbatch -J wd_mse_down_c1_f025_first4 $S train.slurm --run wd_mse_down_c1_f025_first4 --max-ctx 1 --ttt-fast-frac 0.25 --ttt-layers 0,1,2,3 --curriculum 1 --eval-splits curr3 test/n05 test/n10 test/n20 --ttt-loss mse --ttt-params down
sbatch -J wd_mse_down_c1_f025_last4 $S train.slurm --run wd_mse_down_c1_f025_last4 --max-ctx 1 --ttt-fast-frac 0.25 --ttt-layers last4 --curriculum 1 --eval-splits curr3 test/n05 test/n10 test/n20 --ttt-loss mse --ttt-params down
# 本地（GPU 0/4）：c0 下的层位置对照（SWA1 下都卡在 relay，看不出差别）
# python train.py --run wd_c0_f025h8_first4 --ttt-layers 0,1,2,3 --max-ctx 0 ... --gpu 0
# python train.py --run wd_c0_f025h8_last4 --ttt-layers last4 --max-ctx 0 ... --gpu 4
