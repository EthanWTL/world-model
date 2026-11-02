#!/bin/bash
# wave H 续训：注意力 delta 的 run 都在 5000–6000 步才突然学会（原始权重 curr3 .8–1.0），
# EMA（~1000 步的窗口）还停在学会之前，6000 步的评估不公平。同一个 run 名续到 12000 步（从 last.pt 接着跑，lr 是常数）。
# dot 的几个也在慢慢涨，一起续，loss 对比才公平。e2e 5 个变体 6000 步 p 一直是 .20（inner lr 没塌），不续。
cd /project/sds-rise/ethan/world_model/auto_research
R=/project/sds-rise/ethan/Finetune/world/auto_research
ev() { sbatch --dependency=afterok:$2 -J ev_$1 $3 eval.slurm --ckpt $R/$1/ckpt_012000.pt --key ema \
  --splits test/n05 test/n10 test/n20 test/n30 --modes tf free --num 64 --steps 50 --out $R/$1/eval/ckpt_012000_ema.json; }
S="--export=ALL,CODE=$PWD/snapshots/wh,HF_HUB_OFFLINE=1"
C="--max-ctx 0 --curriculum 1 --eval-splits curr3 test/n05 test/n10 test/n20 --ttt-layers all --ttt-params none --steps 12000"
H="--ttt-fast-frac 0.25 --ttt-heads 4"
go() { n=$1; shift; J=$(sbatch --parsable -J ${n}_12k $S train.slurm --run $n $C "$@"); ev $n $J "$S"; echo "$n $J"; }
go wh_v       --ttt-loss mse --ttt-attn v   $H
go wh_k       --ttt-loss mse --ttt-attn k   $H
go wh_q       --ttt-loss mse --ttt-attn q   $H
go wh_qkv     --ttt-loss mse --ttt-attn qkv $H
go wh_v_dot   --ttt-loss dot --ttt-attn v   $H
go wh_qkv_dot --ttt-loss dot --ttt-attn qkv $H
go wh_v_f1    --ttt-loss mse --ttt-attn v
go wh_qkv_f1  --ttt-loss mse --ttt-attn qkv
S="--export=ALL,CODE=$PWD/snapshots/wg,HF_HUB_OFFLINE=1"
C="--max-ctx 0 --curriculum 1 --eval-splits curr3 test/n05 test/n10 test/n20 --steps 12000"
go wg_dot_down_f025h8 --ttt-layers all --ttt-loss dot --ttt-params down --ttt-fast-frac 0.25 --ttt-heads 8
go wg_dot_both_f025h8 --ttt-layers all --ttt-loss dot --ttt-params both --ttt-fast-frac 0.25 --ttt-heads 8
