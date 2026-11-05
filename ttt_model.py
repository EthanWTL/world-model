# TTT 版的因果 DiT：一次前向一个 chunk（960 个 token），state 进、state 出
#   state = 每层窗口里过去 chunk 的 K/V + 每层记忆的 ΔW（memory.py）+ 下一个 chunk 的起始帧号（RoPE）
#   block 里不原地改任何东西，所以每个 block 能做 checkpoint，整段 rollout 能做 BPTT
#   参数名和 model.py 的 WanModel 一样，base 的 EMA 能直接 load；新参数都在 mems.* 下
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from memory import ProjMemory
from model import WanModel, rope_apply, sinusoidal_embedding_1d


def project(attn, p, x, mem, S):
    # 注意力的一个投影（p = "q" / "k" / "v"）加上记忆：P(x) + [0, ΔW·x]；这个投影没开记忆就是原来的 P(x)
    y = getattr(attn, p)(x)
    return y + mem[p].read(S[p], x) if p in mem else y


def attention_chunk(attn, x, grid, freqs, start, ck, cv, mem, S):
    # attn: 这一层的 WanSelfAttention（只用它的权重）；x: [B, T, D] 注意力的输入；grid: (5, 12, 16)
    # start: 这个 chunk 第一帧的帧号（RoPE 的时间位置）；ck / cv: 窗口里过去 chunk 的 K/V [B, n, Tc, d]，None = 不看过去
    # mem: 这一层的记忆 {投影名: ProjMemory}；S: {投影名: ΔW}
    # 返回 (输出 [B, T, D], 这个 chunk 的 K, V [B, n, T, d])，K/V 交给 state 存进窗口
    b, L, _ = x.shape
    n, d = attn.num_heads, attn.head_dim
    q = rope_apply(attn.norm_q(project(attn, "q", x, mem, S)).view(b, L, n, d), grid, freqs, start).transpose(1, 2)
    k = rope_apply(attn.norm_k(project(attn, "k", x, mem, S)).view(b, L, n, d), grid, freqs, start).transpose(1, 2)
    v = project(attn, "v", x, mem, S).view(b, L, n, d).transpose(1, 2)
    K = k if ck is None else torch.cat([ck, k], 2)  # 过去的 K/V 在前，当前 chunk 的在后
    V = v if cv is None else torch.cat([cv, v], 2)
    out = F.scaled_dot_product_attention(q, K, V)  # query 只有当前 chunk 的 960 个
    return attn.o(out.transpose(1, 2).flatten(2)), k, v


def block_chunk(blk, mem, x, e, grid, freqs, start, ck, cv, S, write):
    # 一个 block 处理一个 chunk。blk: WanAttentionBlock；mem: 这一层的记忆（没开 = 空）；x: [B, T, D]
    # e: [B, 1, 6, D] 调制（整个 chunk 共用一份）；S: 这一层的 {投影名: ΔW}；write: 这是 commit（干净 chunk），要写记忆
    # 返回 (x, 这个 chunk 的 K, V, 新的 S)，不原地改任何东西
    shift1, scale1, gate1, shift2, scale2, gate2 = (blk.modulation.unsqueeze(0) + e).unbind(2)
    xa = blk.norm1(x) * (1 + scale1) + shift1  # 注意力的输入，也是记忆写入的 key
    if write:   # commit 这一遍：先用干净 chunk 更新 ΔW，这一遍后面的注意力就读新的（先写后读）；denoise 时只读
        # p -> q, k, v, m -> ProjectMemory()
        S = {p: m.write(S[p], xa, getattr(blk.self_attn, p).weight) for p, m in mem.items()}
    a, k, v = attention_chunk(blk.self_attn, xa, grid, freqs, start, ck, cv, mem, S)
    x = x + a * gate1
    x = x + blk.ffn(blk.norm2(x) * (1 + scale2) + shift2) * gate2
    return x, k, v, S


class State(NamedTuple):
    kv: list    # 每层一个 list：窗口里每个过去 chunk 的 (K, V)，各 [B, n, 960, d]
    mem: list   # 每层一个 dict：{投影名: ΔW [B, nf, D]}，没开记忆的层 = {}
    start: int  # 下一个 chunk 的第一帧是第几帧（RoPE 的时间位置）


class TTTModel(WanModel):
    def __init__(self, proj="q", layers=tuple(range(12)), fast_heads=4, heads=4, loss="delta", eta0=1e-3, renorm=True,
                 max_ctx=0, grad_ckpt=True):
        # proj: 哪些投影开记忆，"q" / "k" / "qk" / "qkv" ...，"" = 都不开；layers: 哪些层开
        # fast_heads: 后面几个注意力 head 做 fast（16 个里），其余兜底；heads: fast 部分切成几块（块对角）
        # max_ctx: 每个 chunk 还能看前面几个 chunk 的 K/V（0 = 只看自己）；grad_ckpt: 训练时每个 block 做 checkpoint
        super().__init__(causal=True)  # init_weights 在这里面跑完；记忆是后加的，它的零初始化不会被 xavier 盖掉
        assert set(proj) <= set("qkv"), proj
        self.max_ctx, self.grad_ckpt = max_ctx, grad_ckpt
        nf = fast_heads * self.blocks[0].self_attn.head_dim  # fast 通道数：4 × 72 = 288
        self.mems = nn.ModuleList([
            nn.ModuleDict({p: ProjMemory(self.dim, nf, heads, loss, eta0, renorm) for p in proj} if i in layers else {})
            for i in range(len(self.blocks))])

    def init_state(self, B):
        # 一个 rollout 开始：窗口是空的，每层的 ΔW 全 0，从第 0 帧开始
        dev = self.patch_embedding.weight.device
        return State([[] for _ in self.blocks], [{p: m.init(B, dev) for p, m in mems.items()} for mems in self.mems], 0)

    def forward(self, x, t, a, state, write=False):
        # 一个 chunk：x [B, 16, 5, 24, 32]，t [B]（σ×1000），a [B] 这个 chunk 的动作；state：前面的 chunk 留下的
        # write=False（denoise）：返回 (速度预测, 原来的 state)；write=True（commit 干净 chunk）：返回 (None, 新的 state)
        if self.freqs.device != x.device:
            self.freqs = self.freqs.to(x.device)
        x = self.patch_embedding(x)                                         # [B, D, 5, 12, 16]
        grid = tuple(x.shape[2:])                                           # (5, 12, 16)
        x = x.flatten(2).transpose(1, 2)                                    # [B, 960, D]
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float()) #[B, time D(256)]
        e = (e + self.action_proj(self.action_embedding(a)))[:, None]      # [B, 1, D]：整个 chunk 共用一份
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))            # [B, 1, 6, D]
        kv, mem = [], []
        for i, blk in enumerate(self.blocks):
            win = state.kv[i]                                               # 窗口里过去 chunk 的 [(K, V), ...]
            ck = torch.cat([k for k, _ in win], 2) if win else None
            cv = torch.cat([v for _, v in win], 2) if win else None
            args = (blk, self.mems[i], x, e0, grid, self.freqs, state.start, ck, cv, state.mem[i], write)
            if self.grad_ckpt and torch.is_grad_enabled():  # 训练：每个 block 只存输入，反传时重算一遍
                x, k, v, S = checkpoint(block_chunk, *args, use_reentrant=False)
            else:
                x, k, v, S = block_chunk(*args)
            kv.append((k, v))
            mem.append(S)
        if not write:  # denoise：只要输出，state 原样返回
            return self.unpatchify(self.head(x, e), grid), state
        keep = lambda w: w[-self.max_ctx:] if self.max_ctx else []          # 只留最近 max_ctx 个；注意 w[-0:] 是整个 list
        return None, State([keep(w + [p]) for w, p in zip(state.kv, kv)], mem, state.start + grid[0])

    def commit(self, x, a, state):
        # 把一个干净的 chunk（训练 = GT，推理 = 刚生成的）σ=0 过一遍：写记忆、K/V 进窗口，返回新的 state
        return self(x, torch.zeros(a.shape[0], device=x.device), a, state, write=True)[1]


def load_base(model, path):
    # 读 base 的 EMA（WanModel 的 state_dict）。只有记忆的参数（mems.*）是新的，保持它们的初始化
    sd = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("mems.") for k in missing), (missing, unexpected)
    return model
