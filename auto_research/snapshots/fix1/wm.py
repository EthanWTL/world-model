# 因果 DiT + FFN fast weights（TTT）。结构、参数名和 world_model/model.py 一样，能直接 load 基础模型的 EMA。
# 和原版的区别：
#   1. 一次只前向一个 chunk（960 个 token）。KV cache 和 fast weights 放在 State 里当输入输出传，block 不原地改东西，
#      所以可以对每个 block 做 checkpoint，整段 rollout 做 BPTT
#   2. RoPE 用 float32 的 cos/sin（原版 float64 复数，A6000/A40 上 FP64 很慢）；角度还是 float64 算的
#   3. TTT：选中的 block 把 FFN 的 up（W1: 1152->4608）和/或 down（W2: 4608->1152）当成每个样本自己的 fast weight，
#      commit 干净 chunk 时按 inner loss 走一步梯度；denoise 只读不写
import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


@dataclass
class TTTConfig:
    layers: tuple = ()      # 哪些 block 做 TTT（空 = 不做）
    params: str = "both"    # 更新哪个矩阵：up / down / both
    loss: str = "dot"       # dot: ℓ = -<f(u), v>（LaCT）；mse: ℓ = ½|f(u) - f_slow(u) - v|²（delta rule）；e2e: 扩散 loss 本身
    value: str = "self"     # v 从哪来：self = Wv·u_t；next = Wv·u（下一个 latent 帧同一位置，In-Place TTT 的"对齐下一个 token"）
                            # read = Wv·u_t + Wr·y_old（这一层用旧权重读出来的 FFN 输出，要 order=rw）：写依赖自己的读 = 显式 self-path
    order: str = "wr"       # wr: 先写后读（LaCT）；rw: 先读后写（In-Place）
    lr: float = 1e-3        # inner lr 初值（softplus 参数化；dot/mse 每个 token 一个，e2e 每层一个标量）
    norm: bool = True       # 每次更新后把每一行的范数拉回 slow weight 的（LaCT 的 weight norm）
    v_init: float = 0.0     # Wv 的初始化 std；0 = 零初始化，一开始什么都不写，模型和 base 完全一样
    e2e_sigma: float = -1.0  # e2e 的 inner loss 用的 σ；<0 = 每次随机抽（和训练一样，shift 3）
    heads: int = 1          # >1：更新量只保留块对角（输入 D/H 一段 <-> hidden F/H 一段），slow weight 还是稠密的（LaCT 多头的类比）


def inv_softplus(y):
    return math.log(math.expm1(y))


def gelu_grad(z):
    # tanh 近似 GELU 的导数（FFN 用的就是它）
    z = z.float()
    k = math.sqrt(2 / math.pi)
    t = torch.tanh(k * (z + 0.044715 * z ** 3))
    return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * k * (1 + 3 * 0.044715 * z * z)


def head_mask(rows, cols, heads, device):
    # [rows, cols] 的块对角 0/1：第 i 行属于 head i // (rows/H)，第 j 列属于 head j // (cols/H)
    return ((torch.arange(rows, device=device)[:, None] * heads // rows) ==
            (torch.arange(cols, device=device)[None, :] * heads // cols)).float()


def renorm(W, ref):
    # 每一行（最后一维）的范数拉回 ref 对应行的范数
    return W * (ref.norm(dim=-1, keepdim=True) / W.norm(dim=-1, keepdim=True).clamp_min(1e-6))


def rope_angles(max_len, dim, theta=10000):
    # 和 model.rope_params 一样，只是返回角度 m·θ_k（float64），不转成复数
    return torch.outer(torch.arange(max_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))


def rope(x, cos, sin):
    # x: [B, L, n, d]，相邻两个数当一个复数转角度；cos/sin: [L, 1, d/2]
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


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.norm_q, self.norm_k = RMSNorm(dim, eps), RMSNorm(dim, eps)

    def forward(self, x, cos, sin, ck, cv):
        # x: [B, L, D]；ck/cv: 窗口里缓存的 K/V [B, n, Lc, d]（None = 没有上下文）
        b, L, _ = x.shape
        n, d = self.num_heads, self.head_dim
        q = rope(self.norm_q(self.q(x)).view(b, L, n, d), cos, sin).transpose(1, 2)
        k = rope(self.norm_k(self.k(x)).view(b, L, n, d), cos, sin).transpose(1, 2)
        v = self.v(x).view(b, L, n, d).transpose(1, 2)
        K = k if ck is None else torch.cat([ck, k], 2)
        V = v if cv is None else torch.cat([cv, v], 2)
        out = F.scaled_dot_product_attention(q, K, V)
        return self.o(out.transpose(1, 2).flatten(2)), k, v


class FastWriter(nn.Module):
    """一个 block 的 TTT 写入（dot / mse）：拿 FFN 的输入 u，按 inner loss 对 W1 / W2 走一步梯度（每个样本一份）"""

    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.v = nn.Linear(dim, dim, bias=False)  # 想写进去的值 v = Wv·u
        if cfg.v_init > 0:
            nn.init.normal_(self.v.weight, std=cfg.v_init)
        else:
            nn.init.zeros_(self.v.weight)
        if cfg.value == "read":
            assert cfg.order == "rw", "value=read 要先用旧权重读"
            self.r = nn.Linear(dim, dim, bias=False)  # 读出来的东西 -> 要写的值，零初始化
            nn.init.zeros_(self.r.weight)
        self.lr = nn.Linear(dim, 2)  # 每个 token 两个 lr（W1、W2 各一个）= softplus(W·u + b)，一开始全是 cfg.lr
        nn.init.zeros_(self.lr.weight)
        nn.init.constant_(self.lr.bias, inv_softplus(cfg.lr))

    def forward(self, u, W1, W2, up, down, y_old=None):
        # u: [B, T, D]；W1: [B, F, D] / W2: [B, D, F]（None = 还是 slow weight）。返回更新后的 (W1, W2)，没更新的原样返回
        # y_old：这一层用旧权重读出来的 FFN 输出（value=read 才用）
        cfg = self.cfg
        T = u.shape[1]
        W1c = up.weight if W1 is None else W1
        W2c = down.weight if W2 is None else W2
        v = self.v(u)
        if cfg.value == "read":
            v = v + self.r(y_old)
        if cfg.value == "next":  # 目标换成下一个 latent 帧同一位置的 token（最后一帧没有下一帧，写 0）
            v = v.unflatten(1, (5, -1))
            v = torch.cat([v[:, 1:], torch.zeros_like(v[:, :1])], 1).flatten(1, 2)
        eta = F.softplus(self.lr(u).float())                       # [B, T, 2]
        if cfg.loss == "dot":
            z = torch.matmul(u, W1c.mT) + up.bias                  # [B, T, F]，用更新前的 W1
            h = F.gelu(z, approximate="tanh")
            g = -v                                                 # ∂ℓ/∂y，ℓ = -<y, v>
        else:
            # mse：只要求 fast 多出来的那部分输出 y - y_slow = v（delta rule）。两个 bf16 matmul 直接相减全是舍入噪声
            # （相对 ~4e-3，会被当成误差写进权重），所以按增量算：z = z0 + ΔW1·u，y - y_slow = W2·(h - h0) + ΔW2·h0
            z0 = up(u).float()
            z = z0 if W1 is None else z0 + torch.matmul(u, (W1 - up.weight).mT).float()
            h0, h = F.gelu(z0, approximate="tanh"), F.gelu(z, approximate="tanh")
            fast = 0 if W1 is None else torch.matmul(h - h0, W2c.mT).float()
            if W2 is not None:
                fast = fast + torch.matmul(h0, (W2 - down.weight).mT).float()
            g = fast - v
        g = g.float() / T                                          # 对 token 取平均
        new1, new2 = W1, W2
        if cfg.params in ("up", "both"):
            dz = torch.matmul(g, W2c).float() * gelu_grad(z)        # [B, T, F] = ∂ℓ/∂z
            G1 = torch.matmul((dz * eta[..., :1]).mT, u)           # [B, F, D]
            if cfg.heads > 1:
                G1 = G1 * head_mask(*G1.shape[-2:], cfg.heads, G1.device)
            new1 = W1c - G1
            if cfg.norm:
                new1 = renorm(new1, up.weight)
        if cfg.params in ("down", "both"):
            G2 = torch.matmul((g * eta[..., 1:]).mT, h)            # [B, D, F]
            if cfg.heads > 1:
                G2 = G2 * head_mask(*G2.shape[-2:], cfg.heads, G2.device)
            new2 = W2c - G2
            if cfg.norm:
                new2 = renorm(new2, down.weight)
        return new1, new2


class BallProbe(nn.Module):
    """从一个 chunk 最后一层的 token 特征读"球现在在几号杯子"：
    5 个杯子所在的列各自平均 -> 共享打分（局部特征：'球在我这儿'）+ 全局平均 -> 5 类（全局编码）"""

    def __init__(self, dim, num_cups=5, grid=(5, 12, 16)):
        super().__init__()
        self.grid = grid
        self.local, self.glob = nn.Linear(dim, 1), nn.Linear(dim, num_cups)
        self.cols = [c.tolist() for c in torch.arange(grid[2]).tensor_split(num_cups)]

    def forward(self, x):
        x = x.float().view(x.shape[0], *self.grid, -1)                                      # [B, f, h, w, D]
        slots = torch.stack([x[:, :, :, c].mean((1, 2, 3)) for c in self.cols], 1)          # [B, 5, D]
        return self.local(slots).squeeze(-1) + self.glob(x.mean((1, 2, 3)))                 # [B, 5]


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
        self.ttt = None  # FastWriter（dot / mse 的层才有）

    def ffn_apply(self, u, W1, W2):
        up, down = self.ffn[0], self.ffn[2]
        z = up(u) if W1 is None else torch.matmul(u, W1.mT) + up.bias
        h = F.gelu(z, approximate="tanh")
        return down(h) if W2 is None else torch.matmul(h, W2.mT) + down.bias

    def forward(self, x, e, cos, sin, ck, cv, W1, W2, write):
        # x: [B, T, D]；e: [B, 1, 6, D]；write = 这是 commit（干净 chunk），TTT 层要更新 fast weight
        # 返回 (x, 这个 chunk 的 K, V, 新 W1, 新 W2)；W 没变就返回 None
        shift1, scale1, gate1, shift2, scale2, gate2 = (self.modulation.unsqueeze(0) + e).unbind(2)
        a, k, v = self.self_attn(self.norm1(x) * (1 + scale1) + shift1, cos, sin, ck, cv)
        x = x + a * gate1
        u = self.norm2(x) * (1 + scale2) + shift2
        new1 = new2 = None
        if write and self.ttt is not None and self.ttt.cfg.order == "wr":
            new1, new2 = self.ttt(u, W1, W2, self.ffn[0], self.ffn[2])
            W1, W2 = new1, new2
        y = self.ffn_apply(u, W1, W2)
        if write and self.ttt is not None and self.ttt.cfg.order == "rw":
            new1, new2 = self.ttt(u, W1, W2, self.ffn[0], self.ffn[2], y)
        return x + y * gate2, k, v, new1, new2


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
    """rollout 的状态：每层窗口里的 K/V（每个 chunk 一份），每层的 fast weight（None = slow），RoPE 的时间起点"""

    def __init__(self, n_layers):
        self.k = [[] for _ in range(n_layers)]
        self.v = [[] for _ in range(n_layers)]
        self.W1 = [None] * n_layers
        self.W2 = [None] * n_layers
        self.start = 0

    def replace(self, **kw):
        s = State.__new__(State)
        s.__dict__.update(self.__dict__)
        s.__dict__.update(kw)
        return s

    def push(self, ks, vs, W1, W2, max_ctx, frames):
        # commit 完：每层把这个 chunk 的 K/V 放进窗口，只留最近 max_ctx 个（None = 全留，0 = 不留）
        # 注意 max(·, 0)：l[len(l) - max_ctx:] 在 len(l) < max_ctx 时是负下标，会只留最后几个（这个 bug 让 max_ctx=7 卡在 3）
        keep = (lambda l: l) if max_ctx is None else (lambda l: l[max(len(l) - max_ctx, 0):] if max_ctx else [])
        return self.replace(k=[keep(o + [k]) for o, k in zip(self.k, ks)], v=[keep(o + [v]) for o, v in zip(self.v, vs)],
                            W1=W1, W2=W2, start=self.start + frames)


class WanTTT(nn.Module):
    def __init__(self, ttt=TTTConfig(), max_ctx=1, grad_ckpt=False, patch_size=(1, 2, 2), in_dim=16, out_dim=16,
                 dim=1152, num_heads=16, ffn_dim=4608, num_layers=12, freq_dim=256, num_actions=13, action_dim=72):
        super().__init__()
        self.ttt, self.max_ctx, self.grad_ckpt = ttt, max_ctx, grad_ckpt
        self.patch_size, self.out_dim, self.dim, self.freq_dim = patch_size, out_dim, dim, freq_dim
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.head = Head(dim, out_dim, patch_size)
        self.blocks = nn.ModuleList([Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        d = dim // num_heads
        self.register_buffer("angles", torch.cat([rope_angles(1024, d - (d // 6) * 4), rope_angles(1024, (d // 6) * 2),
                                                  rope_angles(1024, (d // 6) * 2)], dim=1), persistent=False)  # [1024, 36]
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.action_embedding = nn.Embedding(num_actions, action_dim)
        self.action_proj = nn.Linear(action_dim, dim)
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        if ttt.loss == "e2e":  # e2e：每层每个矩阵一个可学的标量 lr
            self.e2e_lr = nn.Parameter(torch.full((num_layers, 2), inv_softplus(ttt.lr)))
        else:
            for i in ttt.layers:
                self.blocks[i].ttt = FastWriter(dim, ttt)

    def rope_tables(self, start, grid):
        f, h, w = grid
        a = self.angles
        c = a.shape[1]
        at, ah, aw = a[:, :c - 2 * (c // 3)], a[:, c - 2 * (c // 3):c - c // 3], a[:, c - c // 3:]
        ang = torch.cat([at[start:start + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                         ah[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                         aw[:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, c)
        return ang.cos().float(), ang.sin().float()

    def init_state(self):
        return State(len(self.blocks))

    def forward(self, x, t, a, state, write=False, head=True, inner_from=None, inner_train=False, hidden=False):
        # 一个 chunk：x [B, C, 5, H, W]，t [B]（σ×1000），a [B]。返回 (速度预测 or None, 新 state)；write=False 时 state 不变
        # inner_from = l0：这是 e2e 的 inner 前向（只为了对 l0 起的 fast weights 求梯度）
        #   l0 起的层：窗口 K/V 当常数（inner 梯度只走当前 chunk）；训练时用 math attention、不 checkpoint（要二阶梯度）
        #   l0 以下：训练时照常（一阶，checkpoint）；eval 时不建计算图
        x = self.patch_embedding(x)
        grid = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1, 2)                                                 # [B, 960, D]
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e = (e + self.action_proj(self.action_embedding(a)))[:, None]                    # [B, 1, D]：整个 chunk 共用
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))                         # [B, 1, 6, D]
        cos, sin = self.rope_tables(state.start, grid)
        ks, vs, W1s, W2s = [], [], list(state.W1), list(state.W2)
        for i, blk in enumerate(self.blocks):
            ck = (torch.cat(state.k[i], 2) if len(state.k[i]) > 1 else state.k[i][0]) if state.k[i] else None
            cv = (torch.cat(state.v[i], 2) if len(state.v[i]) > 1 else state.v[i][0]) if state.v[i] else None
            inner = inner_from is not None and i >= inner_from
            if inner:
                ck, cv = ck.detach() if ck is not None else None, cv.detach() if cv is not None else None
            use_math = inner and inner_train
            grad = torch.is_grad_enabled() and (inner or inner_from is None or inner_train)
            args = (x, e0, cos, sin, ck, cv, W1s[i], W2s[i], write)
            with torch.set_grad_enabled(grad), (sdpa_kernel(SDPBackend.MATH) if use_math else nullcontext()):
                if self.grad_ckpt and grad and not inner:
                    x, k, v, n1, n2 = checkpoint(blk, *args, use_reentrant=False)
                else:
                    x, k, v, n1, n2 = blk(*args)
            ks.append(k)
            vs.append(v)
            if n1 is not None:
                W1s[i] = n1
            if n2 is not None:
                W2s[i] = n2
        out = self.unpatchify(self.head(x, e), grid) if head else None
        if write:
            state = state.push(ks, vs, W1s, W2s, self.max_ctx, grid[0])
        return (out, state, x) if hidden else (out, state)  # hidden：再返回最后一层的 token 特征 [B, 960, D]（给 probe）

    def e2e_update(self, x0, a, state, sigma, noise):
        # TTT-E2E：inner loss = 这个干净 chunk 自己的扩散 loss（用当前 state 读：窗口 + fast weights），
        # 对 TTT 层的 fast weight 求梯度走一步。训练时 create_graph，外层 loss 能穿过这一步（gradient of gradient）
        cfg, B = self.ttt, x0.shape[0]
        train = torch.is_grad_enabled() and self.training
        W1, W2 = list(state.W1), list(state.W2)
        want = []
        with torch.enable_grad():
            for i in cfg.layers:
                blk = self.blocks[i]
                for Ws, w, p in ((W1, blk.ffn[0].weight, "up"), (W2, blk.ffn[2].weight, "down")):
                    if cfg.params in (p, "both"):
                        cur = w.expand(B, -1, -1) if Ws[i] is None else Ws[i]  # 每个样本一份，梯度才是 per-sample 的
                        if not train:
                            cur = cur.detach().requires_grad_(True)
                        Ws[i] = cur
                        want.append((Ws, i, p))
            s = sigma[:, None, None, None, None]
            x_t = (1 - s) * x0 + s * noise
            target = noise - x0
            l0 = min(cfg.layers)  # 训练：l0 以下的层只要一阶梯度（flash + checkpoint），l0 起要二阶（math，不 checkpoint）
            st = state.replace(W1=W1, W2=W2)
            pred, _ = self.forward(x_t, sigma * 1000, a, st, inner_from=l0, inner_train=train)
            loss = (pred.float() - target).pow(2).mean(dim=(1, 2, 3, 4)).sum()  # 每个样本自己的 loss 加起来
            grads = torch.autograd.grad(loss, [Ws[i] for Ws, i, _ in want], create_graph=train)
        lr = F.softplus(self.e2e_lr.float())
        for (Ws, i, p), g in zip(want, grads):
            blk = self.blocks[i]
            if cfg.heads > 1:
                g = g * head_mask(*g.shape[-2:], cfg.heads, g.device)
            new = Ws[i] - lr[i, 0 if p == "up" else 1] * g
            if cfg.norm:
                new = renorm(new, (blk.ffn[0] if p == "up" else blk.ffn[2]).weight)
            Ws[i] = new if train else new.detach()
        return state.replace(W1=W1, W2=W2)

    def unpatchify(self, x, grid_size):
        B = x.shape[0]
        f, h, w = grid_size
        pt, ph, pw = self.patch_size
        c = self.out_dim
        x = x.view(B, f, h, w, pt, ph, pw, c)
        x = torch.einsum('bfhwpqrc->bcfphqwr', x)
        return x.reshape(B, c, f * pt, h * ph, w * pw)


def commit(model, x0, a, state, sigma_inner=None, noise_inner=None):
    # 把一个干净 chunk 写进 state：e2e 先用 inner loss 更新 fast weights，再 σ=0 前向一遍写 K/V（dot/mse 在 block 里更新）
    if model.ttt.loss == "e2e" and model.ttt.layers:
        state = model.e2e_update(x0, a, state, sigma_inner, noise_inner)
    t = torch.zeros(a.shape[0], device=x0.device)
    _, state = model(x0, t, a, state, write=True, head=False)
    return state


def load_base(model, path):
    # 读基础模型的 EMA（world_model 的 WanModel state_dict），TTT 新加的参数保持初始化
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("ema", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(".ttt." in k or k == "e2e_lr" for k in missing), missing
    return model
