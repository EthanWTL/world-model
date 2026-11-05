# visual memory 方向的模型：因果 DiT（结构、参数名和 world_model/model.py 的 WanModel 一样，尺寸可配），两种前向：
#   forward_seq：整段 episode 一次前向，chunk 因果注意力（训 base：diffusion forcing，每个 chunk 自己的 σ）
#   forward    ：一次一个 chunk（960 个 token），State（KV 窗口 + 每个样本的记忆）当输入输出传，block 不原地改东西，
#                所以每个 block 能 checkpoint、整段 rollout 能 BPTT（和 auto_research/wm.py 一样的做法）
# 记忆 = 原生权重上每个样本自己的增量 ΔW（零初始化 = 和 base 完全一样），commit 干净 chunk 时按 inner loss 走一步，denoise 只读：
#   注意力 q/k/v（AttnMem）：后 fast_heads 个 head 的输出通道 = fast（"slots"），前面的 head 兜底；块对角切 blocks 块
#   FFN down（FFNMem）：方向一的最佳配方（delta、1/4 的 hidden 单元 fast、8 块）
#   写法：delta：ℓ = ½|ΔW·x − t|²（只补差，能覆盖旧的）；dot：每次把 t 整个加上去
#   key：raw = 这一层的输入（方向一）；norm = 每一块的 key 归一化成单位向量、步长 β = 2·sigmoid ∈ (0, 2)：
#        ΔW ← ΔW(I − β·k̂k̂ᵀ) + β·t·k̂ᵀ 的特征值在 (−1, 1]，写多少次都不会发散（方向一的 v 写多了会炸，坑 18）
import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SIZES = {  # dim, heads, ffn, layers
    "200m": (1152, 16, 4608, 12),
    "50m": (640, 10, 2560, 10),
}


@dataclass
class MemConfig:
    proj: str = ""          # 注意力的哪些投影做记忆："q" / "k" / "v" 的任意组合；"" = 不做
    ffn: bool = False       # FFN 的 down 也做记忆
    layers: tuple = ()      # 哪些 block
    fast_heads: int = 4     # 注意力 head 里后几个的通道 fast（slots），其余兜底；= num_heads 就没有兜底
    blocks: int = 4         # 注意力的 fast 部分切几块（块对角）
    ffn_frac: float = 0.25  # FFN hidden 单元里 fast 的比例
    ffn_blocks: int = 8
    loss: str = "delta"     # delta / dot
    key: str = "raw"        # raw / norm
    lr: float = 1e-3        # raw：η 初值（softplus）；norm：β 初值（2·sigmoid）
    renorm: bool = True     # raw：每次写完把 slow + ΔW 每一行的长度拉回 slow 的（方向一）；norm 不用


def inv_softplus(y):
    return math.log(math.expm1(y))


def block_mask(rows, cols, blocks, device=None):
    return ((torch.arange(rows, device=device)[:, None] * blocks // rows) ==
            (torch.arange(cols, device=device)[None, :] * blocks // cols)).float()


def rope_angles(max_len, dim, theta=10000):
    return torch.outer(torch.arange(max_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))


def rope(x, cos, sin):
    # x: [B, L, n, d]，相邻两个数当一个复数转；cos / sin: [L, 1, d/2]（float32，和 model.py 的 float64 复数版一样，快很多）
    xf = x.float().unflatten(-1, (-1, 2))
    x0, x1 = xf[..., 0], xf[..., 1]
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1).flatten(-2).type_as(x)


def sinusoidal_embedding_1d(dim, position):
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_f = x.float()
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_f.type_as(x) * self.weight


class DeltaMem(nn.Module):
    """一个矩阵上的记忆：状态 ΔW [B, out, in]（块对角），读 = x·ΔWᵀ，写 = 对 inner loss 走一步"""

    def __init__(self, n_in, n_out, blocks, cfg):
        super().__init__()
        self.n_in, self.n_out, self.blocks, self.cfg = n_in, n_out, blocks, cfg
        self.register_buffer("mask", block_mask(n_out, n_in, blocks), persistent=False)

    def key(self, x):
        if self.cfg.key == "raw":
            return x
        xb = x.float().unflatten(-1, (self.blocks, -1))
        return (xb / xb.norm(dim=-1, keepdim=True).clamp_min(1e-6)).flatten(-2).type_as(x)

    def read(self, S, x):
        return torch.zeros(*x.shape[:-1], self.n_out, device=x.device, dtype=x.dtype) if S is None \
            else torch.matmul(self.key(x), S.mT.to(x.dtype))

    def write(self, S, x, t, step, slow=None):
        # x: [B, T, in] key 的来源；t: [B, T, out] 目标；step: [B, T, 1] 步长（η 或 β）；slow: [out, in]（renorm 用）
        # 全程 fp32（关掉 autocast）：状态是 fp32，写的量很小，bf16 会把它舍掉（方向一的坑 1 / 13）
        with torch.autocast("cuda", enabled=False):
            k = self.key(x.float())
            err = -t.float() if self.cfg.loss == "dot" else (0 if S is None else torch.matmul(k, S.mT)) - t.float()
            G = torch.matmul((err * step / x.shape[1]).mT, k) * self.mask
            S = -G if S is None else S - G
            if self.cfg.renorm and self.cfg.key == "raw" and slow is not None:
                Ws = slow.float()
                Wf = Ws + S
                S = Wf * (Ws.norm(dim=-1, keepdim=True) / Wf.norm(dim=-1, keepdim=True).clamp_min(1e-6)) - Ws
        return S


class AttnMem(nn.Module):
    """一个 block 的注意力记忆：选中的 q/k/v 各一个 DeltaMem（后 nf 个输出通道），目标 t_p = T_p·a（零初始化）"""

    def __init__(self, dim, num_heads, cfg):
        super().__init__()
        self.cfg = cfg
        self.nf = dim // num_heads * cfg.fast_heads
        self.mem = nn.ModuleDict({p: DeltaMem(dim, self.nf, cfg.blocks, cfg) for p in cfg.proj})
        self.t = nn.ModuleDict({p: nn.Linear(dim, self.nf, bias=False) for p in cfg.proj})
        self.step = nn.Linear(dim, len(cfg.proj))
        for m in self.t.values():
            nn.init.zeros_(m.weight)
        nn.init.zeros_(self.step.weight)
        nn.init.constant_(self.step.bias, inv_softplus(cfg.lr) if cfg.key == "raw" else math.log(cfg.lr / (2 - cfg.lr)))

    def steps(self, a):
        s = self.step(a).float()
        return F.softplus(s) if self.cfg.key == "raw" else 2 * torch.sigmoid(s)

    def write(self, a, mem, attn):
        st = self.steps(a)
        return {p: self.mem[p].write(mem.get(p), a, self.t[p](a), st[..., j:j + 1], getattr(attn, p).weight[-self.nf:])
                for j, p in enumerate(self.cfg.proj)}

    def read(self, p, a, mem):
        # 投影 p 的输出要加的量 [B, T, dim]（前面兜底的通道补 0）
        return F.pad(self.mem[p].read(mem.get(p), a), (a.shape[-1] - self.nf, 0))


class FFNMem(nn.Module):
    """FFN down 的记忆：后 nf 个 hidden 单元到输出的 ΔW2 [B, dim, nf]，key = 这些单元的激活 h，目标 v = W_v·u（零初始化）"""

    def __init__(self, dim, ffn_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.nf = int(round(ffn_dim * cfg.ffn_frac / cfg.ffn_blocks)) * cfg.ffn_blocks
        self.mem = DeltaMem(self.nf, dim, cfg.ffn_blocks, cfg)
        self.v = nn.Linear(dim, dim, bias=False)
        self.step = nn.Linear(dim, 1)
        nn.init.zeros_(self.v.weight)
        nn.init.zeros_(self.step.weight)
        nn.init.constant_(self.step.bias, inv_softplus(cfg.lr) if cfg.key == "raw" else math.log(cfg.lr / (2 - cfg.lr)))

    def write(self, u, h, S, down):
        s = self.step(u).float()
        s = F.softplus(s) if self.cfg.key == "raw" else 2 * torch.sigmoid(s)
        return self.mem.write(S, h[..., -self.nf:], self.v(u), s, down.weight[:, -self.nf:])

    def read(self, h, S):
        return self.mem.read(S, h[..., -self.nf:])


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.norm_q, self.norm_k = RMSNorm(dim, eps), RMSNorm(dim, eps)

    def qkv(self, x, cos, sin, amem=None, mem=None):
        b, L, _ = x.shape
        n, d = self.num_heads, self.head_dim
        out = []
        for p in "qkv":
            y = getattr(self, p)(x)
            if amem is not None and p in amem.cfg.proj:
                y = y + amem.read(p, x, mem)
            out.append(y)
        q = rope(self.norm_q(out[0]).view(b, L, n, d), cos, sin).transpose(1, 2)
        k = rope(self.norm_k(out[1]).view(b, L, n, d), cos, sin).transpose(1, 2)
        return q, k, out[2].view(b, L, n, d).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
        self.amem = None  # AttnMem（只有记忆层有）
        self.fmem = None  # FFNMem

    def forward(self, x, e, cos, sin, ck, cv, mem, write):
        # 一个 chunk：x [B, T, D]，e [B, 1, 6, D]；ck / cv 窗口里的 K/V [B, n, Lc, d]（None = 没有上下文）
        # mem：{"q"/"k"/"v"/"ffn": ΔW}；write = commit（干净 chunk），先写后读（LaCT 的顺序）
        # 返回 (x, 这个 chunk 的 K, V, 新的 mem)
        shift1, scale1, gate1, shift2, scale2, gate2 = (self.modulation.unsqueeze(0) + e).unbind(2)
        xa = self.norm1(x) * (1 + scale1) + shift1
        mem = dict(mem)
        if write and self.amem is not None:
            mem.update(self.amem.write(xa, mem, self.self_attn))
        q, k, v = self.self_attn.qkv(xa, cos, sin, self.amem, mem)
        K = k if ck is None else torch.cat([ck, k], 2)
        V = v if cv is None else torch.cat([cv, v], 2)
        a = self.self_attn.o(F.scaled_dot_product_attention(q, K, V).transpose(1, 2).flatten(2))
        x = x + a * gate1
        u = self.norm2(x) * (1 + scale2) + shift2
        if self.fmem is None:
            y = self.ffn(u)
        else:
            h = self.ffn[1](self.ffn[0](u))
            if write:
                mem["ffn"] = self.fmem.write(u, h, mem.get("ffn"), self.ffn[2])
            y = self.ffn[2](h) + self.fmem.read(h, mem.get("ffn"))
        return x + y * gate2, k, v, mem

    def forward_seq(self, x, e, cos, sin, K, window):
        # 整段：x [B, K·T, D]，e [B, K·T, 6, D]；第 i 个 chunk 的 query 看第 max(0, i - window) .. i 个 chunk（chunk 因果）
        shift1, scale1, gate1, shift2, scale2, gate2 = (self.modulation.unsqueeze(0) + e).unbind(2)
        q, k, v = self.self_attn.qkv(self.norm1(x) * (1 + scale1) + shift1, cos, sin)
        T = q.shape[2] // K
        out = []
        for i in range(K):
            s = 0 if window is None else max(0, i - window)
            out.append(F.scaled_dot_product_attention(q[:, :, i * T:(i + 1) * T], k[:, :, s * T:(i + 1) * T],
                                                      v[:, :, s * T:(i + 1) * T]))
        x = x + self.self_attn.o(torch.cat(out, 2).transpose(1, 2).flatten(2)) * gate1
        return x + self.ffn(self.norm2(x) * (1 + scale2) + shift2) * gate2


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        shift, scale = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).unbind(2)
        return self.head(self.norm(x) * (1 + scale) + shift)


class State:
    """rollout 的状态：每层窗口里的 K/V（每个 chunk 一份）、每层的记忆 {名字: ΔW}、RoPE 的时间起点"""

    def __init__(self, n_layers):
        self.k = [[] for _ in range(n_layers)]
        self.v = [[] for _ in range(n_layers)]
        self.mem = [{} for _ in range(n_layers)]
        self.start = 0

    def push(self, ks, vs, mems, max_ctx, frames):
        # commit 完：每层把这个 chunk 的 K/V 放进窗口，只留最近 max_ctx 个（None = 全留，0 = 不留）
        keep = (lambda l: l) if max_ctx is None else (lambda l: l[max(len(l) - max_ctx, 0):] if max_ctx else [])
        s = State.__new__(State)
        s.k = [keep(o + [k]) for o, k in zip(self.k, ks)]
        s.v = [keep(o + [v]) for o, v in zip(self.v, vs)]
        s.mem, s.start = mems, self.start + frames
        return s


class WanMem(nn.Module):
    def __init__(self, size="200m", mem=MemConfig(), max_ctx=0, grad_ckpt=False, num_actions=3, patch_size=(1, 2, 2),
                 in_dim=16, out_dim=16, freq_dim=256, action_dim=72):
        super().__init__()
        dim, num_heads, ffn_dim, num_layers = SIZES[size]
        self.mem_cfg, self.max_ctx, self.grad_ckpt = mem, max_ctx, grad_ckpt
        self.patch_size, self.out_dim, self.dim, self.freq_dim = patch_size, out_dim, dim, freq_dim
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.head = Head(dim, out_dim, patch_size)
        self.blocks = nn.ModuleList([Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        d = dim // num_heads
        self.register_buffer("angles", torch.cat([rope_angles(1024, d - (d // 6) * 4), rope_angles(1024, (d // 6) * 2),
                                                  rope_angles(1024, (d // 6) * 2)], dim=1), persistent=False)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.action_embedding = nn.Embedding(num_actions, action_dim)
        self.action_proj = nn.Linear(action_dim, dim)
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.init_weights()
        for i in mem.layers:  # 记忆放在 init_weights 之后，它们自己的零初始化不会被盖掉
            if mem.proj:
                self.blocks[i].amem = AttnMem(dim, num_heads, mem)
            if mem.ffn:
                self.blocks[i].fmem = FFNMem(dim, ffn_dim, mem)

    def init_weights(self):
        # 和 world_model/model.py 的 WanModel.init_weights 一样（从头训 base 用）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.head.head.weight)

    def rope_tables(self, start, grid):
        f, h, w = grid
        a = self.angles
        c = a.shape[1]
        at, ah, aw = a[:, :c - 2 * (c // 3)], a[:, c - 2 * (c // 3):c - c // 3], a[:, c - c // 3:]
        ang = torch.cat([at[start:start + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                         ah[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                         aw[:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, c)
        return ang.cos().float(), ang.sin().float()

    def embed(self, t, a):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).float()).view(*t.shape, -1)
        return e + self.action_proj(self.action_embedding(a))

    def init_state(self):
        return State(len(self.blocks))

    def forward(self, x, t, a, state, write=False, head=True):
        # 一个 chunk：x [B, C, 5, H, W]，t [B]（σ×1000），a [B] -> (速度预测 or None, 新 state)；write=False 时 state 不变
        x = self.patch_embedding(x)
        grid = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1, 2)
        e = self.embed(t, a)[:, None]                                                     # [B, 1, D]
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        cos, sin = self.rope_tables(state.start, grid)
        ks, vs, mems = [], [], []
        for i, blk in enumerate(self.blocks):
            ck = (torch.cat(state.k[i], 2) if len(state.k[i]) > 1 else state.k[i][0]) if state.k[i] else None
            cv = (torch.cat(state.v[i], 2) if len(state.v[i]) > 1 else state.v[i][0]) if state.v[i] else None
            args = (x, e0, cos, sin, ck, cv, state.mem[i], write)
            if self.grad_ckpt and torch.is_grad_enabled():
                x, k, v, m = checkpoint(blk, *args, use_reentrant=False)
            else:
                x, k, v, m = blk(*args)
            ks.append(k)
            vs.append(v)
            mems.append(m)
        out = self.unpatchify(self.head(x, e), grid) if head else None
        if write:
            state = state.push(ks, vs, mems, self.max_ctx, grid[0])
        return out, state

    def forward_seq(self, x, t, a, window=None):
        # 整段：x [B, C, 5K, H, W]，t [B, K]（每个 chunk 的 σ×1000），a [B, K] -> 速度预测 [B, C, 5K, H, W]
        x = self.patch_embedding(x)
        grid = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1, 2)
        K = t.shape[1]
        e = self.embed(t, a).repeat_interleave(x.shape[1] // K, dim=1)                   # [B, L, D]
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        cos, sin = self.rope_tables(0, grid)
        for blk in self.blocks:
            if self.grad_ckpt and torch.is_grad_enabled():
                x = checkpoint(blk.forward_seq, x, e0, cos, sin, K, window, use_reentrant=False)
            else:
                x = blk.forward_seq(x, e0, cos, sin, K, window)
        return self.unpatchify(self.head(x, e), grid)

    def unpatchify(self, x, grid_size):
        B = x.shape[0]
        f, h, w = grid_size
        pt, ph, pw = self.patch_size
        c = self.out_dim
        x = x.view(B, f, h, w, pt, ph, pw, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        return x.reshape(B, c, f * pt, h * ph, w * pw)

    def mem_numel(self):
        # 每个样本的记忆：(可写的数 = 块对角里非零的个数, 状态存储 = 整个 ΔW 的个数)
        n_w = n_s = 0
        for blk in self.blocks:
            for m in ([*blk.amem.mem.values()] if blk.amem is not None else []) + ([blk.fmem.mem] if blk.fmem is not None else []):
                n_w += int(m.mask.sum().item())
                n_s += m.n_in * m.n_out
        return n_w, n_s


def commit(model, x0, a, state):
    # 干净 chunk（σ = 0）前向一遍：写 K/V 窗口、写记忆
    _, state = model(x0, torch.zeros(a.shape[0], device=x0.device), a, state, write=True, head=False)
    return state


def parse_layers(s, n):
    if s in ("none", ""):
        return ()
    if s == "all":
        return tuple(range(n))
    if s.startswith("last"):
        return tuple(range(n - int(s[4:]), n))
    if s.startswith("first"):
        return tuple(range(int(s[5:])))
    if s.startswith("every"):
        return tuple(range(0, n, int(s[5:])))
    return tuple(int(i) for i in s.split(","))


def mem_config(args):
    # args：dict（checkpoint 里存的 / argparse 的 vars）
    n = SIZES[args.get("size", "200m")][3]
    return MemConfig(proj=args["mem_proj"], ffn=bool(args["mem_ffn"]), layers=parse_layers(args["mem_layers"], n),
                     fast_heads=args["mem_fast_heads"], blocks=args["mem_blocks"], ffn_frac=args["mem_ffn_frac"],
                     ffn_blocks=args["mem_ffn_blocks"], loss=args["mem_loss"], key=args["mem_key"], lr=args["mem_lr"],
                     renorm=bool(args["mem_renorm"]))


def load_base(model, path):
    # base（train_base.py）的 EMA；记忆新加的参数保持初始化
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("ema", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(".amem." in k or ".fmem." in k for k in missing), missing
    return model
