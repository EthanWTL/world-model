#!/bin/bash
# wave G：三篇论文的做法放进我们的训练配方（c0 + 课程，6000 步）+ 拆开 fallback / head / loss×矩阵 的对照。代码冻结在 snapshots/wg
#   论文：In-Place（dot, 只更新 down, next-frame value, 先读后写, 不归一化；每 6 层一个 = 5,11 / 全部层）
#         LaCT8（dot, both, 8 head, 每 head 144×288 = f0.5, renorm, 先写后读）
#         TTT-E2E（inner loss = 扩散 loss, both, 最后 1/4 层, 一半单元冻结 = static MLP, 不归一化）
#   对照（delta rule, down）：f0.25 单 head / f1 8 head；（f0.25+h8 下）dot down / dot both
#   每个训完自动接完整评估（EMA，n05-n30，tf+free，64 个，50 步）
cd /project/sds-rise/ethan/world_model/auto_research
S="--export=ALL,CODE=$PWD/snapshots/wg,HF_HUB_OFFLINE=1"
R=/project/sds-rise/ethan/Finetune/world/auto_research
C="--max-ctx 0 --curriculum 1 --eval-splits curr3 test/n05 test/n10 test/n20"
run() {  # run <name> <TTT 参数...>：提交训练，训完接完整评估
  n=$1; shift
  J=$(sbatch --parsable -J $n $S train.slurm --run $n $C "$@")
  sbatch --dependency=afterok:$J -J ev_$n $S eval.slurm --ckpt $R/$n/ckpt_006000.pt --key ema \
    --splits test/n05 test/n10 test/n20 test/n30 --modes tf free --num 64 --steps 50 --out $R/$n/eval/ckpt_006000_ema.json
  echo "$n $J"
}
run wg_inplace         --ttt-layers 5,11  --ttt-loss dot --ttt-params down --ttt-value next --ttt-order rw --ttt-norm 0
run wg_inplace_all     --ttt-layers all   --ttt-loss dot --ttt-params down --ttt-value next --ttt-order rw --ttt-norm 0
run wg_lact            --ttt-layers all   --ttt-loss dot --ttt-params both --ttt-fast-frac 0.5 --ttt-heads 8
run wg_e2e             --ttt-layers last3 --ttt-loss e2e --ttt-params both --ttt-fast-frac 0.5 --ttt-norm 0 --ttt-lr 10 --ttt-lr-lr 1e-2 --accum 2
run wg_f025h1          --ttt-layers all   --ttt-loss mse --ttt-params down --ttt-fast-frac 0.25
run wg_f1h8            --ttt-layers all   --ttt-loss mse --ttt-params down --ttt-heads 8
run wg_dot_down_f025h8 --ttt-layers all   --ttt-loss dot --ttt-params down --ttt-fast-frac 0.25 --ttt-heads 8
run wg_dot_both_f025h8 --ttt-layers all   --ttt-loss dot --ttt-params both --ttt-fast-frac 0.25 --ttt-heads 8
# 追加（用户问 loss 本身的影响）：只换 loss = TTT-E2E 的扩散 loss，其余和最佳配方完全一样（down, f0.25+h8, 全部层）。
# 12 层都要二阶图，显存大：只用 B200 + accum 4，时间放宽到 2 天
#   sbatch -C b200 -t 2-00:00:00 ... train.slurm --run wg_e2e_f025h8 $C --ttt-layers all --ttt-loss e2e --ttt-params down --ttt-fast-frac 0.25 --ttt-heads 8 --ttt-lr 10 --ttt-lr-lr 1e-2 --accum 4
# 追加（e2e 设计扫描第一轮）：σ=1（inner loss 只能靠记忆预测整个 chunk）；我们的结构 + both（理论上 e2e 只更新 down 改不动"看不见的"状态）
#   wg_e2e_s1          = wg_e2e + --e2e-sigma 1.0
#   wg_e2e_both_f025h8 = wg_e2e_f025h8 但 --ttt-params both（B200, accum 4, 2 天）
