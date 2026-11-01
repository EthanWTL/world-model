# 剩下的 loss 在哪：tf 设定（chunk 0–6 干净，只给最后一个 chunk 加噪，和训练时 clean prefix 长度 7 一样）
#   1. 最后一个 chunk 的误差里，有多少落在"球可能出现的地方"（5 个杯子下面）
#   2. 把 chunk 0–1 换成另一个 episode 的（只换了球的起始位置，杯子都一样），球那块的误差变不变：不变 = 模型根本没用起始位置
#   python diag_loss.py --ckpts /project/sds-rise/ethan/Finetune/world/shellgame/ema_130000.pt ...
import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from data import ShellGameDataset
from diffusion import add_noise
from eval import VAE, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--split", default="test/n05")
ap.add_argument("--num", type=int, default=64)
ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args()

ds = ShellGameDataset("/project/sds-rise/ethan/Dataset/ShellGame", args.split)
b = default_collate([ds[i] for i in range(args.num)])
x0, a, start = b["latent"], b["actions"], b["ball"][:, 0]
T0 = 5 * (a.shape[1] - 1)  # 最后一个 chunk 的第一个 latent 帧（n05 是 35）

# 球可能出现的地方：GT 解码后最后一个 chunk 里的红色像素，每 4 帧并成一个 latent 帧、8x8 并成一个 latent 像素，往外扩 1 格，
# 再对所有 episode 取并集（球最后在哪个杯子下都算进来）
vae = VAE()
masks = []
for i in range(0, args.num, args.batch):
    f = torch.from_numpy(vae.decode(x0[i:i + args.batch].cuda())[:, 4 * T0 - 3:]).float() / 255  # [b, 20, 192, 256, 3]
    red = ((f[..., 0] - f[..., 1:].amax(-1)) > 0.18).unflatten(1, (5, 4)).any(2).float()  # [b, 5, 192, 256]
    masks.append(F.max_pool2d(F.max_pool2d(red, 8), 3, 1, 1))  # [b, 5, 24, 32]
mask = torch.cat(masks).amax(0)[None, None].cuda()  # [1, 1, 5, 24, 32]
print(f"球可能出现的地方占最后一个 chunk 的 {mask.mean():.1%}", flush=True)
del vae

perm = torch.roll(torch.arange(args.num), 1)  # 每个 episode 配上前一个 episode 的 chunk 0–1
diff = (start != start[perm]).cuda()          # 只看起始位置真的不一样的
ctx = {"真起点": x0, "换起点": torch.cat([x0[perm][:, :, :10], x0[:, :, 10:]], dim=2)}
for ck in args.ckpts:
    model = load_model(ck)
    print(f"== {os.path.basename(os.path.dirname(ck))}/{os.path.basename(ck)}", flush=True)
    for sigma in (0.3, 0.6, 0.9):
        err = {}
        for name, x in ctx.items():
            out = []
            for i in range(0, args.num, args.batch):
                xb, ab = x[i:i + args.batch].cuda(), a[i:i + args.batch].cuda()
                sig = torch.zeros(ab.shape, device="cuda")
                sig[:, -1] = sigma
                torch.manual_seed(i)  # 两种上下文用同一份噪声
                x_t, target = add_noise(xb, sig)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(x_t, sig * 1000, ab)
                out.append((pred.float() - target)[:, :, T0:].pow(2).mean(1, keepdim=True))  # [b, 1, 5, 24, 32]
            err[name] = torch.cat(out)
        e = err["真起点"]
        share = (e * mask).sum() / e.sum()
        inside = (e * mask).sum() / (mask.sum() * len(e))
        outside = (e * (1 - mask)).sum() / ((1 - mask).sum() * len(e))
        w = mask * diff[:, None, None, None, None]
        print(f"  σ={sigma}: 最后一个 chunk 平均误差 {e.mean():.4f} | 落在杯子下面的占 {share:.0%} | "
              f"每个元素：杯子下面 {inside:.4f}，其他地方 {outside:.5f} | "
              f"换起点后杯子下面 {(e * w).sum() / w.sum():.4f} -> {(err['换起点'] * w).sum() / w.sum():.4f}", flush=True)
