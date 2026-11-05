# memory.py 的 playground：每写完一步，这里加一段，最后停在 pdb 里随便看
#   cd /project/sds-rise/ethan/world_model && conda activate world && python playground/play_memory.py
# Step 1–3 只用 CPU（张量都很小），不占卡
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 让 playground/ 里也能 import world_model/ 下的文件
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 必须在 import torch 之前设好

import torch
import torch.nn.functional as F

from memory import ProjMemory, block_mask


# ---- M1-a: block_mask：fast 的 288 个通道（后 4 个 head）× 1152 个输入，切成 4 块 ----
# 预期：形状 (288, 1152)，1 的个数 82944 = 4 块 × 72 行 × 288 列；每行都是 288 个 1
#       第 0 / 71 行（fast head 0）从输入 0..287 写，第 72 行（head 1）288..575，第 287 行（head 3）864..1151
m = block_mask(288, 1152, 4)
print("mask", tuple(m.shape), "1 的个数", int(m.sum()), "每行几个 1", m.sum(1).unique().tolist())
for i in (0, 71, 72, 287):
    cols = m[i].nonzero()
    print(f"第 {i} 行（fast head {i // 72}）从输入的 {cols.min().item()}..{cols.max().item()} 写")


# ---- M1-b: 参数和初值 ----
# 预期：target.weight (288, 1152) 全 0；eta.weight (1, 1152) 全 0；eta.bias (1,) -6.9073 -> η 初值 0.0010
#       state_dict 里只有这三个；mask 是 buffer 但不存（persistent=False，由参数就能算出来）
mem = ProjMemory()
for n, p in mem.named_parameters():
    print(n, tuple(p.shape), "全 0" if not p.any() else f"{p.flatten()[0].item():.4f}")
print("η 初值", f"{F.softplus(mem.eta.bias).item():.4f}")
print("state_dict", list(mem.state_dict()), "| buffers", [n for n, _ in mem.named_buffers()])


# ---- M1-c: init / read ----
# 预期：S (2, 288, 1152) float32 -> 零状态 read (2, 960, 1152)，max 0.0（零状态读出来严格是 0，模型和 base 一样）
#       随机 ΔW：前 864 个通道（兜底的 12 个 head）max 0.0；后 288 个和 einsum 差 0.0（或 1e-7 级，只是求和顺序），量级 0.27
#       autocast 下 read 是 bfloat16（和 P(x) 一样，可以直接相加）
torch.manual_seed(0)
B, T = 2, 960  # 一个 chunk 960 个 token
S = mem.init(B, "cpu")
x = torch.randn(B, T, 1152)
r = mem.read(S, x)
print("S", tuple(S.shape), S.dtype, "| 零状态 read", tuple(r.shape), "max", r.abs().max().item())
S2 = 0.01 * torch.randn(B, 288, 1152)
r2 = mem.read(S2, x)
ref = torch.einsum("btd,bfd->btf", x, S2)  # 第 b 个样本、第 t 个 token：ΔW_b · x_bt
print("兜底通道 max", r2[..., :864].abs().max().item(), "| fast 通道和 einsum 差", (r2[..., 864:] - ref).abs().max().item(),
      f"（量级 {ref.abs().mean().item():.2f}）")
with torch.autocast("cpu", dtype=torch.bfloat16):
    print("autocast 下 read 的 dtype", mem.read(S2, x).dtype)


# ---- M2-a: write 和 autograd 对一下：手写的 G 就是 inner loss 对 ΔW 的梯度 ----
# delta：ℓ = mean_t η_t · ½|ΔW·x_t - t_t|²；dot：ℓ = -mean_t η_t <ΔW·x_t, t_t>；新 ΔW = ΔW - mask ⊙ ∂ℓ/∂ΔW
# target / eta 先随便给个非零的值（零初始化时 delta 和 dot 都只写 0，看不出东西）；renorm 先关，Step 3 再测它
# 预期：delta / dot 和 autograd 的差都是 1e-9 级（我这边 1.9e-9），更新量级 2e-4；块对角外面严格是 0.0
torch.manual_seed(0)
x = torch.randn(B, T, 1152)
W = 0.03 * torch.randn(1152, 1152)  # 假的 slow weight，这一步还用不到（renorm 用）
S = 0.01 * torch.randn(B, 288, 1152)
for loss in ("delta", "dot"):
    mem = ProjMemory(loss=loss, renorm=False)
    torch.nn.init.normal_(mem.target.weight, std=0.02)
    torch.nn.init.normal_(mem.eta.weight, std=0.02)
    S_new = mem.write(S, x, W)
    Sg = S.clone().requires_grad_(True)
    tgt, eta = mem.target(x).float(), F.softplus(mem.eta(x).float())[..., 0]  # [B, T, 288] / [B, T]
    y = x @ Sg.mT  # 旧记忆对每个 token 的回答 ΔW·x_t
    if loss == "delta":
        ell = (eta * 0.5 * (y - tgt).pow(2).sum(-1)).mean(1).sum()  # 对 token 平均，对样本求和（每个样本的梯度互不影响）
    else:
        ell = -(eta * (y * tgt).sum(-1)).mean(1).sum()
    g, = torch.autograd.grad(ell, Sg)
    d = S_new - S
    print(loss, "和 autograd 差", (S_new - (S - g * mem.mask)).abs().max().item(), "| 更新量级", f"{d.abs().max().item():.1e}",
          "| 块对角外", (d * (1 - mem.mask)).abs().max().item())


# ---- M2-b: 同一个 key 写两次：delta 读回第二个目标（覆盖），dot 读回两个之和（累加）----
# 小例子：D = 8，后 4 个通道 fast，不分块，η = 1，一个 token，key x1 长度为 1
# 把 target 设成 t·x1ᵀ，这样 T·x1 = t。delta：ΔW·x1 ← ΔW·x1 - (ΔW·x1 - t)(x1ᵀx1) = t，之前存的被整个换掉
# 预期：delta 第 1 次读回 t1、第 2 次读回 t2；dot 第 1 次 t1、第 2 次 t1+t2（数字对照第一行打印的）；兜底通道一直是 0.0
def fmt(v):
    return [round(a, 3) for a in v.tolist()]


x1 = F.normalize(torch.randn(1, 1, 8), dim=-1)
t1, t2 = torch.randn(4), torch.randn(4)
print("t1", fmt(t1), "| t2", fmt(t2), "| t1+t2", fmt(t1 + t2))
for loss in ("delta", "dot"):
    mem8 = ProjMemory(dim=8, nf=4, heads=1, loss=loss, eta0=1.0, renorm=False)
    S8 = mem8.init(1, "cpu")
    for i, t in enumerate((t1, t2)):
        with torch.no_grad():
            mem8.target.weight.copy_(torch.outer(t, x1[0, 0]))
        S8 = mem8.write(S8, x1, torch.randn(8, 8))
        r = mem8.read(S8, x1)[0, 0]
        print(loss, f"写第 {i + 1} 次后读回", fmt(r[4:]), "| 兜底通道", r[:4].abs().max().item())


# ---- M2-c: 训练是 bf16 autocast：新 ΔW 还是 fp32；旧的 S 没被原地改（checkpoint 重算时不会写两次）----
# 预期：torch.float32 | True
mem = ProjMemory(renorm=False)
torch.nn.init.normal_(mem.target.weight, std=0.02)
S0 = S.clone()
with torch.autocast("cpu", dtype=torch.bfloat16):
    S_new = mem.write(S, x, W)
print("autocast 下新 ΔW", S_new.dtype, "| 旧的 S 没被改", torch.equal(S, S0))


# ---- M3-a: renorm 开：零初始化写完还是严格的 0（模型和 base 一样）----
# 预期：0.0（target 是 0 -> G = 0 -> Wf = Ws，比例严格是 1，Ws·1 - Ws = 0）
mem = ProjMemory()  # 默认：delta、renorm 开、target 零初始化
print("零初始化写完 max |ΔW|", mem.write(mem.init(B, "cpu"), x, W).abs().max().item())


# ---- M3-b: 从 0 开始写一次：renorm 关，行长度变了；renorm 开，行长度严格等于 slow 的，但块对角外面也被缩放了 ----
# η 调到 1，让一次写的量看得清。默认 1e-3 时一次只变 5e-5；但 auto 训完以后 ΔW 能到 W 的 10–16%（log 里的 dW_q），那时 renorm 就要紧了
# 预期：renorm False：行长度/slow 1.064762..1.162405，块对角外 0.0
#       renorm True ：行长度/slow 1.000000..1.000000，块对角外 1.9e-02（整行一起缩放，所以 ΔW 要存整块 [288, 1152]）
for renorm in (False, True):
    torch.manual_seed(1)
    mem = ProjMemory(eta0=1.0, renorm=renorm)
    torch.nn.init.normal_(mem.target.weight, std=0.02)
    S_new = mem.write(mem.init(B, "cpu"), x, W)
    ratio = (W[-288:] + S_new).norm(dim=-1) / W[-288:].norm(dim=-1)  # [B, 288]：写完每一行的长度 / slow 的
    print(f"renorm {renorm}: 行长度/slow {ratio.min().item():.6f}..{ratio.max().item():.6f} | "
          f"块对角外 max {(S_new * (1 - mem.mask)).abs().max().item():.1e}")

breakpoint()
