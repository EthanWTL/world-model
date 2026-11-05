#!/bin/bash
# 魔方（2026-11-03，用户："先做数据 然后数据做好了就开始用一个比较promising的setting开始从头train"）：
#   1. 等本地编码 worker（datagen/build_rubik.py --encode --claim，GPU 2/3 各两个）领完所有 shard、退出
#   2. 不抢 shard 再扫一遍，补齐缺的 latent（被打断的 shard；正常情况下什么都不做）
#   3. 冻结代码到 snapshots/rubik，base 从头训：GPU 2/3 DDP，整段 chunk 因果注意力（有历史的上限，也是记忆微调的起点），40000 步
#   GPUS=2,3 setsid nohup bash waves/rubik_base.sh > /project/sds-rise/ethan/Finetune/world/auto_research/visual/logs/rubik_base.log 2>&1 &
cd /project/sds-rise/ethan/world_model/auto_research/visual
PY=/project/sds-rise/ethan/miniforge3/envs/world/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
while pgrep -f "build_rubik.py --encode --claim" > /dev/null; do sleep 300; done
echo "$(date '+%F %T') encode workers done; sweeping for missing latents"
(cd datagen && CUDA_VISIBLE_DEVICES=${GPUS%,*} $PY -u build_rubik.py --encode) || exit 1
for s in train test/n00 test/n02 test/n04 test/n08 test/n12 test/n16 test/n24; do
  echo "$s: $(wc -l < /project/sds-rise/ethan/Dataset/RubikMemory/$s/meta.jsonl) episodes, $(find /project/sds-rise/ethan/Dataset/RubikMemory/$s/latents -name '*.npy' | wc -l) latents"
done
CODE=$PWD/snapshots/rubik
mkdir -p $CODE/datagen && cp *.py $CODE/ && cp datagen/*.py $CODE/datagen/
echo "$(date '+%F %T') base starts on GPUs $GPUS (code frozen in $CODE)"
cd $CODE && CUDA_VISIBLE_DEVICES=$GPUS $PY -m torch.distributed.run --standalone --nproc_per_node 2 -- \
  train_base.py --task rubik --run rubik_base --steps 40000
