#!/bin/bash
# wave H：把记忆放进注意力的 q/k/v 投影（AttnWriter）。代码冻结在 snapshots/wh
#   都是 c0 + 课程，6000 步，12 层全做，FFN 不动（--ttt-params none）
#   兜底 + head：f0.25 = 16 个注意力 head 里后 4 个的 q/k/v 通道是 fast，前 12 个兜底；h4 = 每个 fast head 只从输入的 1/4 段写（块对角）
#   目标 t_P = T_P·a（"这个 chunk 希望 P 往哪移"，零初始化）；delta：补 ΔW_P·a 和 t_P 的差；dot：每次整个加上 t_P
#   A 哪个投影：v / k / q / qkv（delta）  B loss：dot v / dot qkv / e2e qkv  C 兜底+head：f1 单 head 的 v / qkv
#   e2e：q/k 的二阶梯度很冲（inner lr 初值 10 时 gn 到 3e6），初值改 1（可学）
cd /project/sds-rise/ethan/world_model/auto_research
S="--export=ALL,CODE=$PWD/snapshots/wh,HF_HUB_OFFLINE=1"
R=/project/sds-rise/ethan/Finetune/world/auto_research
C="--max-ctx 0 --curriculum 1 --eval-splits curr3 test/n05 test/n10 test/n20 --ttt-layers all --ttt-params none"
run() {  # run <name> [sbatch 参数 --] <TTT 参数...>：提交训练，训完接完整评估
  n=$1; shift; X=()
  if [ "$1" = "-X" ]; then X=($2); shift 2; fi
  J=$(sbatch --parsable "${X[@]}" -J $n $S train.slurm --run $n $C "$@")
  sbatch --dependency=afterok:$J -J ev_$n $S eval.slurm --ckpt $R/$n/ckpt_006000.pt --key ema \
    --splits test/n05 test/n10 test/n20 test/n30 --modes tf free --num 64 --steps 50 --out $R/$n/eval/ckpt_006000_ema.json
  echo "$n $J"
}
H="--ttt-fast-frac 0.25 --ttt-heads 4"
run wh_v        --ttt-loss mse --ttt-attn v   $H
run wh_k        --ttt-loss mse --ttt-attn k   $H
run wh_q        --ttt-loss mse --ttt-attn q   $H
run wh_qkv      --ttt-loss mse --ttt-attn qkv $H
run wh_v_dot    --ttt-loss dot --ttt-attn v   $H
run wh_qkv_dot  --ttt-loss dot --ttt-attn qkv $H
run wh_qkv_e2e  -X "-t 1-12:00:00" --ttt-loss e2e --ttt-attn qkv $H --ttt-lr 1 --ttt-lr-lr 1e-2 --accum 4
run wh_v_f1     --ttt-loss mse --ttt-attn v
run wh_qkv_f1   --ttt-loss mse --ttt-attn qkv
