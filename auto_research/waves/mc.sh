#!/bin/bash
# wave MC（2026-11-02，用户："不同的 state 容量和参数容量的联系"）：state 数 × 参数容量 × 训练时间 的 grid
#   state：k = 1 / 2 / 3 / 5 个球（Dataset/ShellGameMulti）→ 5 / 20 / 60 / 120 种状态（k=4 也是 120，先不跑）
#   参数：q 的 fast 行数 r = 16 / 72 / 288 / 1152（4 块）→ 每层 4.6K / 20.7K / 83K / 332K 个数；288 = wh_q 配方
#   时间：每 1000 步快评（k{K}/curr3、n05、n10 的 tf，32 个，原始权重），曲线 = 时间轴；20000 步，4000 一存
#   其余就是 wh_q：delta、q、全部 12 层、课程、NorMuon 1e-4、B=8；从单球基础模型 ema_200000 直接微调
#   完全没有窗口（max_ctx 0，训练和评估都是）：记忆是 chunk 之间唯一的通路
#   训完自动完整评估 20000 步的 EMA（n05–n30，tf + free，64 个，50 步）
#   用法（先 cp *.py snapshots/mc/）：
#     MODE=slurm KS="1 2" RS="16 72" SEEDS=0 bash waves/mc.sh
#     MODE=local GPU=1 KS="1" RS="16 72 288 1152" SEEDS=0 bash waves/mc.sh     （本地：同一张卡上并排起几个）
cd /project/sds-rise/ethan/world_model/auto_research
CODE=$PWD/snapshots/mc
R=/project/sds-rise/ethan/Finetune/world/auto_research
BASE=/project/sds-rise/ethan/Finetune/world/shellgame_normuon/ema_200000.pt
STEPS=${STEPS:-20000}
C="--data multi --base $BASE --max-ctx 0 --curriculum 1 --ttt-layers all --ttt-loss mse --ttt-params none --ttt-attn q
   --ttt-heads 4 --steps $STEPS --keep-every 4000 --eval-every 1000 --eval-num 32 --workers 3"
PY=/project/sds-rise/ethan/miniforge3/envs/world/bin/python
for k in ${KS:-1 2 3 5}; do for r in ${RS:-16 72 288 1152}; do for s in ${SEEDS:-0 1}; do
  n=mbc_k${k}_r${r}_s$s
  f=$(python3 -c "print($r / 1152)")
  T="--run $n $C --balls $k --ttt-fast-frac $f --seed $s --eval-splits k$k/curr3 k$k/test/n05 k$k/test/n10"
  E="--ckpt $R/$n/ckpt_$(printf %06d $STEPS).pt --key ema --balls $k --modes tf free --num 64 --steps 50
     --out $R/$n/eval/ckpt_$(printf %06d $STEPS)_ema.json"
  T=$(echo $T); E=$(echo $E)  # 压成一行：sh -c "..." 里的换行会把命令截成两条（第一次本地启动就这样丢了后半截参数）
  if [ "$MODE" = local ]; then
    mkdir -p $R/logs
    (cd $CODE && CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
      nohup sh -c "$PY -u train.py $T --gpu 0 && $PY -u evaluate.py $E --gpu 0" >> $R/logs/${n}_local.log 2>&1 &)
    # >>：停了再起（train.py 从 last.pt 续）时不冲掉之前的输出
    echo "$n local GPU $GPU"
  else
    S="--export=ALL,CODE=$CODE,HF_HUB_OFFLINE=1"
    J=$(sbatch --parsable -J $n --constraint="a100|a40|a6000|b200|rtxpro6000" --time=2-00:00:00 $S train.slurm $T)
    sbatch --dependency=afterok:$J -J ev_$n $S eval.slurm $E > /dev/null
    echo "$n slurm $J"
  fi
done; done; done
