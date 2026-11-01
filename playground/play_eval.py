# eval.py 的 playground：每写完一步，这里加一段，最后停在 pdb 里随便看
#   cd /project/sds-rise/ethan/world_model && conda activate world && python playground/play_eval.py
# 默认用 GPU 0（训练也在上面跑，会稍微拖慢训练）
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 让 playground/ 里也能 import world_model/ 下的文件
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 必须在 import torch 之前设好
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # Wan2.1 VAE 已经在 ~/.cache/huggingface 里，不用联网

import numpy as np
import torch
from torch.utils.data import default_collate

from data import ShellGameDataset
from eval import VAE, load_model, save_mp4
from rollout import rollout
from shellgame.detect import ball_slot

RUN = Path("/project/sds-rise/ethan/Finetune/world/shellgame")

# ---- E1: 生成 2 个 episode -> 解码 -> 存成「左 GT | 右生成」的 mp4 -> 认球 ----
model = load_model(RUN / "last.pt", key="model")  # 训练早期 EMA 还没跟上，先用原始参数
vae = VAE()
ds = ShellGameDataset("/project/sds-rise/ethan/Dataset/ShellGame", "test/n05")  # 5 次交换 -> 8 个 chunk
batch = default_collate([ds[i] for i in range(2)])
x0, a = batch["latent"].cuda(), batch["actions"].cuda()

t0 = time.time()
with torch.autocast("cuda", dtype=torch.bfloat16):
    video = rollout(model, x0[:, :, :10], a, steps=50, max_ctx=7)  # 给前 2 个 chunk（揭开、盖上），生成后面 6 个
print(f"生成 {tuple(video.shape)}，用了 {time.time() - t0:.0f} s")

gen, gt = vae.decode(video), vae.decode(x0)  # [2, 157, 192, 256, 3] uint8
print("解码后", gen.shape, gen.dtype)
for i in range(2):
    save_mp4(RUN / "samples" / f"step005000_n05_{i}.mp4", np.concatenate([gt[i], gen[i]], axis=2))  # 左 GT，右生成
    print(f"episode {i}: 球最后应该在 {batch['ball'][i, -1].item()} | GT 视频里认出 {ball_slot(gt[i][-1])} | 生成的视频里认出 {ball_slot(gen[i][-1])}")
print("视频在", RUN / "samples")

breakpoint()
