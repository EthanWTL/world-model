#!/bin/bash
# 第一轮（2026-11-03）：哪种原生权重记忆能存住具体的视觉内容（旋转方块四个面的随机色块），多少 slot、多少块、什么写法
#   base：从头训（train_base.py，diffusion forcing，整段 chunk 因果注意力，2 卡，20000 步）
#   之后每个 run：从 base 的 ema_020000 出发，max_ctx 0（关掉历史 KV、没有首帧重注入），12 层全做，12000 步，B=8，
#     训练 gap ~ U{0,2,4,6}，NorMuon 1e-4，EMA .999；训完自动完整评估（EMA，gap 0–32，tf + free，64 个，50 步，存视频）
#   A 投影（fast 4 个 head = 288 行，4 块，delta）：ctrl / v / q / qkv / k / ffn / qk / qv / kv（ffn = 方向一配方：down，1/4 单元，8 块）
#   B 写法（v、qkv）：归一化 key + β∈(0,2)（写多少次都不发散）；v 换成 dot（只累加）
#   C slot（v）：fast head 1 / 8 / 16（16 = 没有兜底）    D 块（v）：1 / 16
#   顺序 = 优先级（排队时先上的先跑）
#   用法：bash waves/r1.sh            （先冻结代码到 snapshots/r1，再交 base，run 都 afterok base，全在 Slurm）
#         SKIP_BASE=1 bash waves/r1.sh  （base 已经训好：只交评估和 run，不等）
#         LOCAL="2:r1_v 3:r1_q" ...      （本地队列：点名的 run 不交 Slurm，排在本地队列最前面；其余照样交 Slurm 兜底。
#                                         本地用点名里出现的卡（每张卡同时 1 个训练 + 1 个评估），base 的两个评估也在本地跑；
#                                         队列 = $R/r1_queue.tsv，waves/local_queue.py 每分钟重读，Slurm 上还在排队的它会取消、拿回本地跑）
#   2026-11-03：Slurm 上的 base 排了 13 小时没排上，取消了，改成本地两张卡（waves/r1_local.sh，训完自动调这个脚本）；
#     Slurm 整体堵住（账号 fairshare 0.08，94 个在排），LOCAL 从"每张卡固定跑两个"改成上面的本地队列
set -e
cd /project/sds-rise/ethan/world_model/auto_research/visual
CODE=$PWD/snapshots/r1
if [ ! -d $CODE ]; then mkdir -p $CODE/datagen && cp *.py $CODE/ && cp datagen/*.py $CODE/datagen/; fi
R=/project/sds-rise/ethan/Finetune/world/auto_research/visual
S="--export=ALL,CODE=$CODE,HF_HUB_OFFLINE=1"
PY=/project/sds-rise/ethan/miniforge3/envs/world/bin/python
RUNS=(  # 名字|参数；顺序 = 优先级
  "r1_ctrl|"
  "r1_v|--mem-proj v"
  "r1_q|--mem-proj q"
  "r1_qkv|--mem-proj qkv"
  "r1_k|--mem-proj k"
  "r1_ffn|--mem-ffn 1"
  "r1_v_norm|--mem-proj v --mem-key norm --mem-lr 0.5"
  "r1_qkv_norm|--mem-proj qkv --mem-key norm --mem-lr 0.5"
  "r1_v_dot|--mem-proj v --mem-loss dot"
  "r1_v_f1|--mem-proj v --mem-fast-heads 1"
  "r1_v_f8|--mem-proj v --mem-fast-heads 8"
  "r1_v_f16|--mem-proj v --mem-fast-heads 16"
  "r1_v_b1|--mem-proj v --mem-blocks 1"
  "r1_v_b16|--mem-proj v --mem-blocks 16"
  "r1_qk|--mem-proj qk"
  "r1_qv|--mem-proj qv"
  "r1_kv|--mem-proj kv"
)
args_of() { for r in "${RUNS[@]}"; do if [ "${r%%|*}" = "$1" ]; then echo "${r#*|}"; fi; done; return 0; }
DEP=""
if [ -z "$SKIP_BASE" ]; then
  B=$(sbatch --parsable -J v_base $S train_base.slurm --run base --steps 20000)
  echo "base $B"
  DEP="--dependency=afterok:$B"
fi
if [ -z "$LOCAL" ]; then  # 全交 Slurm
  for c in -1 0; do  # base 自己：KV 全留（有历史的上限）/ 什么都不看（没训过的"无跨时间记忆"）
    sbatch $DEP -J ev_base_ctx$c $S eval.slurm --ckpt $R/base/ema_020000.pt --max-ctx $c > /dev/null
  done
  for r in "${RUNS[@]}"; do
    n=${r%%|*}; J=$(sbatch --parsable $DEP -J $n $S train.slurm --run $n ${r#*|}); echo "$n $J"
  done
  exit 0
fi
# 本地队列：名字 \t train|eval \t 完成标志文件 \t 参数
Q=$R/r1_queue.tsv
declare -A ON=()
GPUS=""
ORDER=""
for x in $LOCAL; do
  g=${x%%:*}; n=${x#*:}; ON[$n]=1; ORDER+="$n "
  if [[ " $GPUS " != *" $g "* ]]; then GPUS+="$g "; fi
done
for r in "${RUNS[@]}"; do n=${r%%|*}; if [ -z "${ON[$n]}" ]; then ORDER+="$n "; fi; done
{
  for c in -1 0; do
    printf "ev_base_ctx%s\teval\t%s\t%s\n" $c $R/base/eval/ema_020000_ema_ctx$c.json "--ckpt $R/base/ema_020000.pt --max-ctx $c"
  done
  for n in $ORDER; do
    printf "%s\ttrain\t%s\t%s\n" $n $R/$n/eval/ckpt_012000_ema.json "--run $n $(args_of $n)"
  done
} > $Q
mkdir -p $R/logs
PYTHONDONTWRITEBYTECODE=1 setsid nohup $PY -u waves/local_queue.py --gpus $GPUS --queue $Q \
  >> $R/logs/local_queue.log 2>&1 < /dev/null &
echo "local queue on GPUs $GPUS: $Q ($(wc -l < $Q) items), log $R/logs/local_queue.log"
# 2026-11-03 16:30 用户把任务换成魔方，旋转方块这轮不再往 Slurm 交（SLURM_FALLBACK=1 才交）；本地队列先跑着当参考，魔方数据要卡时停
if [ -z "$SLURM_FALLBACK" ]; then exit 0; fi
for r in "${RUNS[@]}"; do  # 没点名的照样交 Slurm 兜底：Slurm 先排上就在那边跑，本地轮到时还在排队就被本地拿走
  n=${r%%|*}
  if [ -n "${ON[$n]}" ]; then continue; fi
  J=$(sbatch --parsable $DEP -J $n $S train.slurm --run $n ${r#*|}) || J="sbatch failed"
  echo "$n $J"
done
