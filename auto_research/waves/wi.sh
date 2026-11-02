#!/bin/bash
# wave I（2026-11-01 晚）：
#   1) 用户问论文 LaCT 的"用 k 预测 v"：我们的写入 key 一直就是这一层自己的输入（k = u / a），加 --ttt-key proj（k = x + W_k·x，可学，只用来写）
#   2) 12000 步时注意力 q / k 单独做记忆外推得很好（free N20 .97、N30 .88–.91），比 FFN 最好的还好 → 换 seed 确认
#   3) 公平对比：FFN 最佳配方也续到 12000 步（拷 wd_mse_down_c0_f025_h8 的 last.pt 到 wi_ffn_12k 接着跑，原 run 不动）
cd /project/sds-rise/ethan/world_model/auto_research
S="--export=ALL,CODE=$PWD/snapshots/wi,HF_HUB_OFFLINE=1"
R=/project/sds-rise/ethan/Finetune/world/auto_research
C="--max-ctx 0 --curriculum 1 --eval-splits curr3 test/n05 test/n10 test/n20 --ttt-layers all"
FFN="--ttt-loss mse --ttt-params down --ttt-fast-frac 0.25 --ttt-heads 8"
ATT="--ttt-loss mse --ttt-params none --ttt-fast-frac 0.25 --ttt-heads 4"
run() {  # run <name> <steps> <TTT 参数...>：提交训练，训完接完整评估（最后一步的 EMA）
  n=$1; st=$2; shift 2
  J=$(sbatch --parsable -J $n $S train.slurm --run $n $C --steps $st "$@")
  sbatch --dependency=afterok:$J -J ev_$n $S eval.slurm --ckpt $R/$n/ckpt_$(printf %06d $st).pt --key ema \
    --splits test/n05 test/n10 test/n20 test/n30 --modes tf free --num 64 --steps 50 --out $R/$n/eval/ckpt_$(printf %06d $st)_ema.json
  echo "$n $J"
}
run wi_ffn_key 6000  $FFN --ttt-key proj
run wi_q_key   12000 $ATT --ttt-attn q --ttt-key proj
run wi_ffn_12k 12000 $FFN
run wi_q_s1    12000 $ATT --ttt-attn q --seed 1
run wi_q_s2    12000 $ATT --ttt-attn q --seed 2
run wi_k_s1    12000 $ATT --ttt-attn k --seed 1
