# train.py 的 playground：每写完一步，这里加一段，最后停在 pdb 里随便看
#   cd /project/sds-rise/ethan/world_model && conda activate world && python playground/play_train.py
# 默认用 GPU 0（0/1 是我们自己的卡，VAE 编码也在上面跑）；换卡：CUDA_VISIBLE_DEVICES=1 python playground/play_train.py
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 让 playground/ 里也能 import world_model/ 下的文件
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 必须在 import torch 之前设好

import torch

from data import ShellGameDataset, infinite_loader
from diffusion import diffusion_forcing_loss
from train import build, get_args, setup, train_step

# ---- 3: get_args + setup + build ----
args = get_args([])
rank, world = setup(args.seed)
model, ema, opt = build(args, world)
print(f"GPU {os.environ['CUDA_VISIBLE_DEVICES']} | rank {rank}/{world} | model {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M params")

ds = ShellGameDataset(args.root, "train")
batch = next(infinite_loader(ds, batch_size=2, num_workers=0))
x0, a = batch["latent"].cuda(), batch["actions"].cuda()
t = torch.full(a.shape, 500.0, device="cuda")  # 每个 chunk 一个 timestep（0 = 干净，1000 = 纯噪声）
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    v = model(x0, t, a)
print(f"forward: x0 {tuple(x0.shape)} -> v {tuple(v.shape)} {v.dtype}, |v| max {v.abs().max():.1f}（head 初始化为 0，一开始输出全是 0）")
print(f"GPU 显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")

# ---- 4: train_step + update_ema ----
args.warmup = 1  # playground 里不 warmup，马上就能看到 loss 往下走；正式训练用默认的 1000
batch = next(infinite_loader(ds, batch_size=2, num_workers=0))  # 8 张卡时每张卡就是 16 // 8 = 2
w0 = next(ema.parameters()).clone()
torch.cuda.reset_peak_memory_stats()
losses = []
for step in range(50):  # 同一个 batch 反复训：loss 会一直降（过拟合），说明梯度传通了
    t0 = time.time()
    loss, gn = train_step(model, ema, opt, batch, step, args)
    losses.append(loss)
    if step % 10 == 9:
        print(f"step {step + 1:2d} | loss {sum(losses[-10:]) / 10:.3f} | grad_norm {gn:.2f} | "
              f"lr {opt.param_groups[0]['lr']:.1e} | {time.time() - t0:.2f} s/step")
print(f"EMA 50 步只挪了 {(next(ema.parameters()) - w0).abs().max():.1e} | 显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")

# ---- 4 拆开看：backward 只算梯度，opt.step 才改参数 ----
p = model.head.head.weight  # 拿一个参数来看
w = p.detach().clone()
opt.zero_grad(set_to_none=True)
print("zero_grad 之后：p.grad =", p.grad)
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = diffusion_forcing_loss(model, x0, a)
loss.backward()
print(f"backward 之后：p.grad 范数 {p.grad.norm():.2e}，参数变了 {(p - w).abs().max():.1e}")
opt.step()
print(f"opt.step 之后：参数变了 {(p - w).abs().max():.1e}")

breakpoint()
