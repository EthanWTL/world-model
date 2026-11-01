# normuon.py 的 playground：每写完一步，这里加一段，最后停在 pdb 里随便看
#   cd /project/sds-rise/ethan/world_model && conda activate world && python playground/play_normuon.py
# NM-1、NM-2 只用 CPU；NM-3 要一张空卡（约 24 GB，默认 GPU 0，先 nvidia-smi 看一下）：
#   CUDA_VISIBLE_DEVICES=<空卡> python playground/play_normuon.py
import io
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 让 playground/ 里也能 import world_model/ 下的文件
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 必须在 import torch 之前设好

import torch
from torch.utils.data import default_collate

from normuon import COEFFS, NorMuon, orthogonalize

# ---- NM-1a: 只看一个奇异值 s：每一步变成 a·s + b·s³ + c·s⁵ ----
# 6 列分别是一开始的 s = 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1（先除以 1.02）
# 预期第 5 步：0.097 0.847 0.997 1.016 1.086 0.904。中间几步会冲过 1，没关系，5 步是一起设计的；只有 1e-4 太小，拉不上来
s = torch.tensor([1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0], dtype=torch.float64) / 1.02
for k, (a, b, c) in enumerate(COEFFS):
    s = a * s + b * s**3 + c * s**5
    print(f"第 {k + 1} 步:", " ".join(f"{x:.3f}" for x in s.tolist()))


# ---- NM-1b: 一个强方向 + 一个弱 100 倍的方向：正交化之后两个方向一样大 ----
# 三种形状就是我们模型里的 Linear：q/k/v/o [1152, 1152]、FFN 升维 [4608, 1152]、FFN 降维 [1152, 4608]
# 预期：每种形状都是 1.00 / 0.01 -> 0.93 / 0.99，输出 bfloat16、形状和输入一样
torch.manual_seed(0)
for m, n in [(1152, 1152), (4608, 1152), (1152, 4608)]:
    U, _ = torch.linalg.qr(torch.randn(m, 2))  # 两个互相垂直的单位向量（列）
    V, _ = torch.linalg.qr(torch.randn(n, 2))
    G = 1.0 * torch.outer(U[:, 0], V[:, 0]) + 0.01 * torch.outer(U[:, 1], V[:, 1])
    O = orthogonalize(G)
    w = [(O.float() * torch.outer(U[:, i], V[:, i])).sum().item() for i in range(2)]  # O 在两个方向上各有多大
    print(f"{m}x{n}: 强/弱方向 1.00 / 0.01 -> {w[0]:.2f} / {w[1]:.2f}，{O.dtype}，{tuple(O.shape)}")


# ---- NM-1c: 满秩的随机矩阵：输出的奇异值都在 1 附近 ----
# 预期：输入奇异值 0.0..67.4 / 34.3..101.9 / 34.0..101.4 -> 中位数 ≈ 1，10%–90% ≈ 0.87–1.13
# （1152x1152 里有几个奇异值接近 0，正交化之后还是很小，不影响中位数）
torch.manual_seed(0)
for m, n in [(1152, 1152), (4608, 1152), (1152, 4608)]:
    G = torch.randn(m, n)
    sv_in, sv = torch.linalg.svdvals(G), torch.linalg.svdvals(orthogonalize(G).float())
    print(f"{m}x{n}: 输入奇异值 {sv_in.min():.1f}..{sv_in.max():.1f} -> "
          f"输出中位数 {sv.median():.2f}，10%–90% {sv.quantile(0.1):.2f}–{sv.quantile(0.9):.2f}")


# ---- NM-2a: AdamW 那一组和 torch.optim.AdamW 一样（同样的初值、同样的梯度，走 5 步）----
# 预期：1e-07 左右（两边写法不同，只差浮点舍入）
torch.manual_seed(0)
others = [torch.randn(1152, requires_grad=True), torch.randn(1152, 16, 1, 2, 2, requires_grad=True)]  # 像 bias、Conv3d 的 weight
mat = torch.randn(64, 32, requires_grad=True)
ref = [p.detach().clone().requires_grad_() for p in others]
opt = NorMuon([mat], others, lr=1e-3)
ref_opt = torch.optim.AdamW(ref, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
for _ in range(5):
    for p, q in zip(others, ref):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
    mat.grad = torch.randn_like(mat)
    opt.step()
    ref_opt.step()
print("NM-2a 和 torch AdamW 的 max |diff|:", max((p - q).abs().max().item() for p, q in zip(others, ref)))


# ---- NM-2b: 矩阵那一组：整体 RMS ≈ 0.2·lr，而且每一行（每个输出神经元）的步子一样大 ----
# 故意让梯度每一行的大小差 100 倍（logspace 1e-2..1），像有的神经元梯度大、有的小
# W 从 0 开始、不加 weight decay，所以走一步之后的 W 就是这一步的更新量
# 预期：RMS/lr 0.186 / 0.201 / 0.199；行 RMS 最大/最小：只做正交化（Muon）2.7 / 33.2 / 1.6，NorMuon 都是 1.00
torch.manual_seed(0)
for m, n in [(1152, 1152), (4608, 1152), (1152, 4608)]:
    W = torch.zeros(m, n, requires_grad=True)
    opt = NorMuon([W], [], lr=1e-3, weight_decay=0.0)
    W.grad = torch.logspace(-2, 0, m)[:, None] * torch.randn(m, n)
    muon = orthogonalize(W.grad).float().pow(2).mean(1).sqrt()  # 只做正交化时每一行的 RMS
    opt.step()
    rows = W.detach().pow(2).mean(1).sqrt()
    print(f"{m}x{n}: RMS/lr {W.detach().pow(2).mean().sqrt() / 1e-3:.3f} | "
          f"行 RMS 最大/最小：Muon {muon.max() / muon.min():.1f}，NorMuon {rows.max() / rows.min():.2f}")


# ---- NM-2c: 续训：state_dict 存下来再读回去，接着走一步，两边完全一样 ----
# 像 train.py 那样经过 torch.save / torch.load（同一个进程里直接 load_state_dict 的话，两个优化器会共用同一份 state 张量）
# 预期：0.0；矩阵存 m [256, 128]、v [256, 1]，bias 存 step 4、m [256]、v [256]
torch.manual_seed(0)
W1, b1 = torch.randn(256, 128, requires_grad=True), torch.randn(256, requires_grad=True)
opt1 = NorMuon([W1], [b1])
for _ in range(3):
    W1.grad, b1.grad = torch.randn_like(W1), torch.randn_like(b1)
    opt1.step()
buf = io.BytesIO()
torch.save(opt1.state_dict(), buf)
buf.seek(0)
W2, b2 = W1.detach().clone().requires_grad_(), b1.detach().clone().requires_grad_()
opt2 = NorMuon([W2], [b2])
opt2.load_state_dict(torch.load(buf))
g, gb = torch.randn_like(W1), torch.randn_like(b1)
W1.grad, b1.grad, W2.grad, b2.grad = g, gb, g.clone(), gb.clone()
opt1.step()
opt2.step()
print("NM-2c 读回来再走一步的 max |diff|:", max((W1 - W2).abs().max().item(), (b1 - b2).abs().max().item()))
shapes = lambda st: {k: tuple(v.shape) if torch.is_tensor(v) else v for k, v in st.items()}
print("矩阵的 state:", shapes(opt2.state[W2]), "| bias 的 state:", shapes(opt2.state[b2]))


# ---- NM-3: train.py 的 --opt normuon：真模型分组，用 train_step 走一步 ----
# 先加载 110K 的 EMA：新初始化的模型 head 的权重是 0，第一步只有 head 有梯度，看不出别的矩阵怎么更新
# 预期：NorMuon 77 个 200.84M，AdamW 117 个 0.32M
#   RMS(ΔW)/lr：(64, 1152) 0.200；(1152, 1152) 0.02..0.14；(4608, 1152) 0.08..0.16；(1152, 4608) 0.06..0.16；
#               (1152, 256) 0.05；(1152, 72) 0.09；(6912, 1152) 0.025；AdamW 那 117 个 0.96
#   opt.step 大约 40 ms（训练一步约 1 s）
# 为什么矩阵大多低于 0.2：第一步的动量就是一个 batch（2 个样本）的梯度，秩很低。比如 time_projection [6912, 1152]
#   的输入只有 2×8 = 16 个不同的 (t, a)，梯度的秩 ≤ 16，正交化只能把这 16 个方向拉到 1，RMS ≈ 0.2·√(16/1152) ≈ 0.024。
#   动量攒了很多步之后方向变多，会往 0.2 靠。AdamW 第一步就是 lr·sign(g)，所以 ≈ 1
from data import ShellGameDataset
from train import build, get_args, train_step

args = get_args(["--opt", "normuon"])
model, ema, opt = build(args, 1)
for g in opt.param_groups:
    print(f"{'NorMuon' if g['muon'] else 'AdamW'}: {len(g['params'])} 个，{sum(p.numel() for p in g['params']) / 1e6:.2f}M")
model.load_state_dict(torch.load("/project/sds-rise/ethan/Finetune/world/shellgame/ema_110000.pt", map_location="cpu"))
ds = ShellGameDataset(args.root, "train")
batch = default_collate([ds[i] for i in range(2)])
before = {p: p.detach().clone() for p in model.parameters()}
torch.manual_seed(0)
loss, gn = train_step(model, ema, opt, batch, args.warmup, args)  # step = warmup：lr 已经是满的 2e-4
print(f"loss {loss:.4f}，grad_norm {gn:.2f}")
for g in opt.param_groups:
    if g["muon"]:
        by_shape = defaultdict(list)
        for p in g["params"]:
            by_shape[tuple(p.shape)].append((p.detach() - before[p]).pow(2).mean().sqrt().item() / g["lr"])
        for s, r in by_shape.items():
            print(f"  {s}: {len(r)} 个，RMS(ΔW)/lr {min(r):.3f}..{max(r):.3f}")
    else:
        d = torch.cat([(p.detach() - before[p]).flatten() for p in g["params"]])
        print(f"  AdamW 那 {len(g['params'])} 个：RMS(ΔW)/lr {d.pow(2).mean().sqrt().item() / g['lr']:.3f}")
for _ in range(2):  # train_step 走完梯度还在，再 step 两次只为了计时（第一次要建 state，会慢一点）
    torch.cuda.synchronize()
    t0 = time.time()
    opt.step()
    torch.cuda.synchronize()
    print(f"opt.step {1000 * (time.time() - t0):.0f} ms")

breakpoint()
