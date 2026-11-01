#!/bin/bash
# wave C：课程训练（N~U{0..5}），按 toy 的结果排优先级（delta rule 最好）。代码冻结在 snapshots/wc。
# 每行就是一个 sbatch；本地两行（GPU 0/4）手动跑。c0 = max_ctx 0（不看过去的 chunk），c1 = SWA1
cd /project/sds-rise/ethan/world_model/auto_research
S="--export=ALL,CODE=$PWD/snapshots/wc"
C="--curriculum 1 --eval-splits curr3 test/n05 test/n10 test/n20"
# 本地 GPU0：python train.py --run wc_mse_down_c0 --max-ctx 0 --ttt-layers all --ttt-loss mse --ttt-params down $C --gpu 0
# 本地 GPU4：python train.py --run wc_mse_down_c1 --max-ctx 1 --ttt-layers all --ttt-loss mse --ttt-params down $C --gpu 4
sbatch --hold -J wc_mse_both_c1 $S train.slurm --run wc_mse_both_c1 --max-ctx 1 --ttt-layers all --ttt-loss mse $C
sbatch --hold -J wc_ctrl_c1 $S train.slurm --run wc_ctrl_c1 --max-ctx 1 $C
sbatch --hold -J wc_dot_up_c0 $S train.slurm --run wc_dot_up_c0 --max-ctx 0 --ttt-layers all --ttt-params up $C
sbatch --hold -J wc_dot_both_c1 $S train.slurm --run wc_dot_both_c1 --max-ctx 1 --ttt-layers all $C
sbatch --hold -J wc_dot_down_c1 $S train.slurm --run wc_dot_down_c1 --max-ctx 1 --ttt-layers all --ttt-params down $C
sbatch --hold -J wc_e2e_last3_c1 $S train.slurm --run wc_e2e_last3_c1 --max-ctx 1 --ttt-layers last3 --ttt-loss e2e --ttt-lr 10 --ttt-lr-lr 1e-2 --accum 2 $C
sbatch --hold -J wc_mse_both_c0 $S train.slurm --run wc_mse_both_c0 --max-ctx 0 --ttt-layers all --ttt-loss mse $C
sbatch -J wc_read_down_c1 $S train.slurm --run wc_read_down_c1 --max-ctx 1 --ttt-layers all --ttt-order rw --ttt-value read --ttt-params down $C  # 加于 toy 结果之后
