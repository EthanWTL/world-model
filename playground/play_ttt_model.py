# ttt_model.py 的 playground：每写完一步，这里加一段，最后停在 pdb 里随便看
#   cd /project/sds-rise/ethan/world_model && conda activate world && python playground/play_ttt_model.py
# Step 4、5a 只用 CPU（半分钟左右）；5b 要 GPU：先 nvidia-smi，挑一张空卡或者我们自己任务在用的卡
#   CUDA_VISIBLE_DEVICES=<卡> python playground/play_ttt_model.py
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 让 playground/ 里也能 import world_model/ 下的文件
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 必须在 import torch 之前设好

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import default_collate

from data import ShellGameDataset
from memory import ProjMemory
from model import WanModel
from ttt_model import TTTModel, block_chunk, load_base

# 随机初始化的模型就够了：这里测的是"和原版算的一样"，不需要训练好的权重
torch.manual_seed(0)
base = WanModel(causal=True)
blk, freqs = base.blocks[0], base.freqs
B, grid = 2, (5, 12, 16)  # 一个 chunk：5 个 latent 帧 × 12 × 16 个 patch
T = 5 * 12 * 16           # 960 个 token
x = torch.randn(B, T, 1152)
e = 0.1 * torch.randn(B, 1, 6, 1152)  # 调制：整个 chunk 共用一份（原版是每个 token 一份，值都一样）


# ---- 4-a: 不开记忆：block_chunk == 原来的 WanAttentionBlock（一个 chunk、不看过去）----
# 预期：差 0.0（量级 0.80）；K、V 都是 (2, 16, 960, 72)；S 是 {}
with torch.no_grad():
    ref = blk(x, e.expand(B, T, 6, 1152), grid, freqs, None, None, 0, False)  # 原版：e 每个 token 一份
    out, k, v, S = block_chunk(blk, {}, x, e, grid, freqs, 0, None, None, {}, False)
print("4-a 和原版差", (out - ref).abs().max().item(), f"（量级 {ref.abs().mean().item():.2f}）",
      "| K", tuple(k.shape), "V", tuple(v.shape), "| S", S)


# ---- 4-b: 开 q 记忆、零初始化、write=True：输出和不开记忆严格一样，ΔW 写完还是 0 ----
# 预期：差 0.0；写完 ΔW max 0.0
mem = nn.ModuleDict({"q": ProjMemory()})
S0 = {"q": mem["q"].init(B, "cpu")}
with torch.no_grad():
    out_m, _, _, S1 = block_chunk(blk, mem, x, e, grid, freqs, 0, None, None, S0, True)
print("4-b 和不开记忆差", (out_m - out).abs().max().item(), "| 写完 ΔW max", S1["q"].abs().max().item())


# ---- 4-c: 先写后读：write=True 的输出 == 拿写完的新 ΔW 再只读跑一遍 ----
# target 给个非零值，让记忆真的写东西
# 预期：差 0.0；ΔW max 1.7e-04；只读时 S 原样返回 True；输出和不开记忆差 3.3e-05（η = 1e-3，写得很少）
torch.nn.init.normal_(mem["q"].target.weight, std=0.02)
with torch.no_grad():
    out_w, _, _, S1 = block_chunk(blk, mem, x, e, grid, freqs, 0, None, None, S0, True)
    out_r, _, _, S_same = block_chunk(blk, mem, x, e, grid, freqs, 0, None, None, S1, False)
print("4-c write=True 的输出 vs 拿新 ΔW 只读", (out_w - out_r).abs().max().item(), f"| ΔW max {S1['q'].abs().max().item():.1e}",
      "| 只读时 S 原样返回", S_same is S1, f"| 输出和不开记忆差 {(out_w - out).abs().max().item():.1e}")


# ---- 4-d: 看过去的 chunk：先跑 chunk 0 拿到它的 K/V，再跑 chunk 1（从第 5 帧开始）带上 chunk 0 的 K/V
#      == 原版把两个 chunk 拼起来做因果注意力，取后半段 ----
# 预期：差 0.0；不带过去的 K/V 时差 0.05（说明 chunk 1 确实看到了 chunk 0）
x2 = torch.randn(B, T, 1152)  # 第二个 chunk
with torch.no_grad():
    ref2 = blk(torch.cat([x, x2], 1), e.expand(B, 2 * T, 6, 1152), (10, 12, 16), freqs, (2, None), None, 0, False)[:, T:]
    _, k0, v0, _ = block_chunk(blk, {}, x, e, grid, freqs, 0, None, None, {}, False)
    out2, _, _, _ = block_chunk(blk, {}, x2, e, grid, freqs, 5, k0, v0, {}, False)
    alone, _, _, _ = block_chunk(blk, {}, x2, e, grid, freqs, 5, None, None, {}, False)
print("4-d 带上一个 chunk 的 K/V vs 原版两个 chunk 一起跑", (out2 - ref2).abs().max().item(),
      f"| 不带过去的差 {(alone - ref2).abs().max().item():.2f}")


# ---- 5a-a: 建模型：新参数只有记忆的 target / eta，都在 mems.* 下 ----
# 预期：proj 'q'：新参数 3,995,148（12 层 × 332,929），总 205.16M；'qkv'：11,985,444，总 213.15M；''：0，总 201.17M（= base）
#       mems[0] 是一个 ModuleDict，里面只有 q 的 ProjMemory（target 1152->288，eta 1152->1）
for proj in ("q", "qkv", ""):
    m = TTTModel(proj=proj)
    new = sum(p.numel() for n, p in m.named_parameters() if n.startswith("mems."))
    print(f"proj '{proj}': 新参数 {new:,} | 总 {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")
model = TTTModel()
print(model.mems[0])


# ---- 5a-b: load_base：base 的 194 个参数全部对上，只缺 36 个 mems.*（12 层 × 3 个），记忆还是零初始化 ----
# 预期：base 194 个 | 缺 36 ['mems.0.q.eta.bias', 'mems.0.q.eta.weight', 'mems.0.q.target.weight'] | 多 []
#       q 的权重和 ckpt 一样 True | target 还是全 0 True | η 0.0010
BASE = "/project/sds-rise/ethan/Finetune/world/shellgame_normuon/ema_160000.pt"
load_base(model, BASE)
sd = torch.load(BASE, map_location="cpu")
missing, unexpected = sorted(set(model.state_dict()) - set(sd)), sorted(set(sd) - set(model.state_dict()))
print("base", len(sd), "个 | 缺", len(missing), missing[:3], "| 多", unexpected)
print("q 的权重和 ckpt 一样", torch.equal(model.blocks[0].self_attn.q.weight, sd["blocks.0.self_attn.q.weight"]),
      "| target 还是全 0", not model.mems[0]["q"].target.weight.any(),
      f"| η {F.softplus(model.mems[0]['q'].eta.bias).item():.4f}")


# ---- 5a-c: init_state：窗口空、每层 ΔW 全 0、从第 0 帧开始；没开记忆的层是 {} ----
# 预期：kv 是 12 个空 list | mem[0] {'q': (2, 288, 1152)} max 0.0 | start 0
#       qk、只开 0/4/8 层、16 个 head 全 fast、不分块（auto 的 f1 消融）：
#       [['q', 'k'], [], [], [], ['q', 'k'], [], [], [], ['q', 'k'], [], [], []]，ΔW (1, 1152, 1152)，mask 全是 1（1327104 = 1152²）
st = model.init_state(2)
print("kv", st.kv, "| mem[0]", {p: tuple(s.shape) for p, s in st.mem[0].items()}, "max", st.mem[0]["q"].abs().max().item(),
      "| start", st.start)
m2 = TTTModel(proj="qk", layers=(0, 4, 8), fast_heads=16, heads=1)
st2 = m2.init_state(1)
print([list(d) for d in st2.mem], {p: tuple(s.shape) for p, s in st2.mem[0].items()},
      "| mask 1 的个数", int(m2.mems[0]["q"].mask.sum()))


# ======== 5b（GPU）：全部 fp32、关掉 TF32，看的是"算的一样"，不是速度 ========
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ep = default_collate([ShellGameDataset("/project/sds-rise/ethan/Dataset/ShellGame", "test/n05")[0]])  # 一个 N=5 的 episode
x0, a = ep["latent"].cuda(), ep["actions"].cuda()  # [1, 16, 40, 24, 32]：8 个 chunk
K = a.shape[1]
z = torch.zeros(1, device="cuda")  # σ = 0
base = WanModel(causal=True)
base.load_state_dict(sd)  # sd：5a-b 读进来的 base EMA
base = base.cuda().eval()
mq = load_base(TTTModel(proj="q", max_ctx=7), BASE).cuda().eval()  # 零初始化的 q 记忆；窗口 7 = 能看到前面所有 chunk
m0 = load_base(TTTModel(proj="", max_ctx=7), BASE).cuda().eval()   # 不开记忆


def per_chunk(m):
    # 逐 chunk：用前面 commit 进去的 state 算 chunk c 的输出（σ = 0），再把 chunk c commit 进去
    st, outs = m.init_state(1), []
    for c in range(K):
        xc = x0[:, :, 5 * c:5 * c + 5]
        y, st_same = m(xc, z, a[:, c], st)
        assert st_same is st  # denoise 不改 state
        outs.append(y)
        st = m.commit(xc, a[:, c], st)
    return torch.cat(outs, 2), st


def per_c(d):  # 每个 chunk 的最大差
    return [f"{d[:, :, 5 * c:5 * c + 5].abs().max().item():.1e}" for c in range(d.shape[2] // 5)]


# ---- 5b-a: 逐 chunk（state 进 state 出）== base 把 8 个 chunk 拼起来一次前向（σ 全是 0，因果注意力）----
# 预期：逐 chunk 每个 chunk 差 2e-4 / 3e-5 级（量级 0.82）。这是 fp32 舍入在 12 层里放大，不是错：
#       base 自己"只跑 chunk 0"和"8 个一起跑的 chunk 0"数学上一模一样，也差 1.8e-4（同一行前半段）
with torch.no_grad():
    full = base(x0, torch.zeros(1, K, device="cuda"), a)
    alone0 = base(x0[:, :, :5], torch.zeros(1, 1, device="cuda"), a[:, :1])
    out_q, st_q = per_chunk(mq)
    out_0, _ = per_chunk(m0)
print("5b-a base 自己：只跑 chunk 0 vs 一起跑", f"{(alone0 - full[:, :, :5]).abs().max().item():.1e}",
      "| 逐 chunk vs 一起跑", per_c(out_q - full), f"（量级 {full.abs().mean().item():.2f}）")


# ---- 5b-b: 零初始化的 q 记忆 == 不开记忆（严格）；8 个 chunk 写完 ΔW 还是 0；state 往后走对了 ----
# 预期：差 0.0 | ΔW max 0.0 | start 40 | 窗口 7（max_ctx = 7，8 次 commit 只留最近 7 个）
print("5b-b 零初始化 q 记忆 vs 不开", (out_q - out_0).abs().max().item(), "| 8 个 chunk 写完 ΔW max",
      max(s["q"].abs().max().item() for s in st_q.mem), "| start", st_q.start, "| 窗口", len(st_q.kv[0]))


# ---- 5b-c: checkpoint 开和关，梯度一样（teacher forcing 跑 3 个 chunk：commit 0 -> denoise 1 -> commit 1 -> denoise 2）----
# 预期：两次 loss 一样（1.5389）；梯度最大差 1e-6 ~ 1e-5 级，每次跑都不太一样（梯度最大值 1.75；反传里有不确定的 kernel，
#       只是舍入）；230 个参数都有梯度；激活显存 4.9 -> 1.2 GB（这里只有 3 个 chunk、B=1；8 个 chunk、B=8 时差得更多）
#       target 的梯度不是 0（层 0 / 11 都是 2e-3 左右）；eta 的梯度严格是 0：零初始化时什么都没写（err = 0），η 多大都一样
model = load_base(TTTModel(proj="q", max_ctx=0), BASE).cuda().train()
g = torch.Generator(device="cuda").manual_seed(0)
C = 3
sig = torch.rand(C, generator=g, device="cuda")
noise = torch.randn(1, 16, 5 * C, 24, 32, generator=g, device="cuda")  # 两次用同样的 σ 和噪声


def grads(ckpt):
    model.grad_ckpt = ckpt
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    m_before = torch.cuda.memory_allocated()
    st, loss = model.init_state(1), 0
    for c in range(C):
        xc, nc = x0[:, :, 5 * c:5 * c + 5], noise[:, :, 5 * c:5 * c + 5]
        if c >= 1:  # denoise chunk c：读 state，算 loss
            pred, _ = model((1 - sig[c]) * xc + sig[c] * nc, (sig[c] * 1000).view(1), a[:, c], st)
            loss = loss + (pred - (nc - xc)).pow(2).mean()
        if c < C - 1:  # commit chunk c：写记忆
            st = model.commit(xc, a[:, c], st)
    loss.backward()
    act = (torch.cuda.max_memory_allocated() - m_before) / 2 ** 30
    return loss.item(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}, act


l1, g1, act1 = grads(False)
l2, g2, act2 = grads(True)
print("5b-c loss", f"{l1:.4f} {l2:.4f}", "| 梯度最大差", f"{max((g1[n] - g2[n]).abs().max().item() for n in g1):.1e}",
      f"| 梯度最大值 {max(v.abs().max().item() for v in g1.values()):.2f}", "| 有梯度的参数", len(g1), len(g2),
      f"| 激活显存 {act1:.1f} -> {act2:.1f} GB")
print("     target 梯度 层 0 / 11",
      f"{g1['mems.0.q.target.weight'].norm().item():.1e} / {g1['mems.11.q.target.weight'].norm().item():.1e}",
      "| eta 梯度", g1["mems.0.q.eta.weight"].norm().item())


# ---- 5b-d: K/V 窗口：commit 3 个 chunk 以后窗口里留几个 ----
# 预期：max_ctx 0 -> 0 个（训练用的设置，只看自己）；1 -> 1 个；7 -> 3 个（还没满）；start 都是 15
for mc in (0, 1, 7):
    model.max_ctx = mc
    with torch.no_grad():
        st = model.init_state(1)
        for c in range(3):
            st = model.commit(x0[:, :, 5 * c:5 * c + 5], a[:, c], st)
    print(f"5b-d max_ctx {mc}: 窗口里 {len(st.kv[0])} 个 chunk，start {st.start}")

breakpoint()
