# 跟踪诊断：同一批 episode、同样的初始噪声，逐个 checkpoint 看最后揭开时球在哪个杯子下
#   free：给 2 个 GT chunk（揭开、盖上），后面全部自己生成（和 eval.py 一样）
#   tf  ：除了最后一个 chunk 全给 GT，只生成最后揭开的那一个（没有生成误差累积，只看跟踪）
# 除了准确率，还看生成的球和「交换 k 次之后球在哪」一致的比例：只跟住前几次交换的模型，小 k 上会高
#   python diag.py --ckpts /project/sds-rise/ethan/Finetune/world/shellgame/ema_100000.pt ...
# 每个 checkpoint 存一个 json：<ckpt 所在目录>/eval/diag_n05/<ckpt 名>.json
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import default_collate

from data import ShellGameDataset
from eval import VAE, load_model
from rollout import rollout
from shellgame.detect import ball_slot

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--split", default="test/n05")
ap.add_argument("--num", type=int, default=64)
ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args()

ds = ShellGameDataset("/project/sds-rise/ethan/Dataset/ShellGame", args.split)
num = min(args.num, len(ds))
ball = np.array([ep["ball"] for ep in ds.eps[:num]])  # [E, K]：每个 chunk 结束时球在哪，ball[:, 1 + k] = 交换 k 次之后
N = ball.shape[1] - 3                                # K = 揭开 + 盖上 + N 次交换 + 揭开
print("只跟住前 k 次交换时的准确率（k = 0..N）:",
      " ".join(f"{np.mean(ball[:, 1 + k] == ball[:, -1]):.2f}" for k in range(N + 1)), flush=True)
vae = VAE()
for ck in args.ckpts:
    model = load_model(ck)
    torch.manual_seed(0)  # 每个 checkpoint 的初始噪声都一样，差别只来自权重
    gen = {"free": [], "tf": []}
    for i in range(0, num, args.batch):
        b = default_collate([ds[j] for j in range(i, min(i + args.batch, num))])
        x0, a = b["latent"].cuda(), b["actions"].cuda()
        for mode, n_ctx in (("free", 2), ("tf", a.shape[1] - 1)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                video = rollout(model, x0[:, :, :5 * n_ctx], a, steps=50, max_ctx=7)
            gen[mode] += [ball_slot(v[-1]) for v in vae.decode(video)]
    print(f"== {Path(ck).stem}", flush=True)
    for mode, g in gen.items():
        g = np.array(g)
        after = [np.mean(g == ball[:, 1 + k]) for k in range(N + 1)]
        hist = np.bincount(g + 1, minlength=6)  # -1（没认出球），0..4
        print(f"  {mode:4s}: acc {after[-1]:.2f} | 和交换 0..{N} 次后的位置一致 {' '.join(f'{x:.2f}' for x in after)} "
              f"| 没认出球 {hist[0]} | 落在 0-4 号 {hist[1:].tolist()}", flush=True)
    out = Path(ck).parent / "eval" / f"diag_{Path(args.split).name}" / f"{Path(ck).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"ckpt": ck, "split": args.split, "ball": ball.tolist(), "gen": gen}, open(out, "w"))
