# 玩具版 Shell Game：只留"SWA1 + 每个 chunk commit 时写 FFN fast weights、下一个 chunk 读"这个结构，几分钟一个配置，
# 用来快速验证哪种更新规则能跟踪不断变化的状态（LOG.md 的分析 A）。
#   每个 chunk = 5 个 token（5 个杯子位置）。chunk 0：哪个位置有球；之后每个 chunk：一次交换（被换的两个位置标上对方是谁）；
#   最后一个 chunk：问球在哪（每个位置一个 logit，5 选 1）。训练 N=5，测 N=5/10/20/30。
#   python toy.py --ttt-layers all --ttt-params both --ttt-loss dot --layers 4
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm import FastWriter, TTTConfig

PAIRS = torch.tensor([(i, j) for i in range(5) for j in range(i + 1, 5)])


def episodes(B, N, device):
    start = torch.randint(0, 5, (B,), device=device)
    swaps = PAIRS.to(device)[torch.randint(0, 10, (B, N), device=device)]  # [B, N, 2]
    pos, ar = start.clone(), torch.arange(B, device=device)
    for k in range(N):
        a, b = swaps[:, k, 0], swaps[:, k, 1]
        pos = torch.where(pos == a, b, torch.where(pos == b, a, pos))
    return start, swaps, pos


class Block(nn.Module):
    def __init__(self, dim, heads, ffn_dim):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.o = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.heads, self.ttt = heads, None

    def forward(self, x, ck, cv, W1, W2, write):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        K = k if ck is None else torch.cat([ck, k], 2)
        V = v if cv is None else torch.cat([cv, v], 2)
        x = x + self.o(F.scaled_dot_product_attention(q, K, V).transpose(1, 2).flatten(2))
        u = self.n2(x)
        if write and self.ttt is not None and self.ttt.cfg.order == "wr":
            W1, W2 = self.ttt(u, W1, W2, self.ffn[0], self.ffn[2])
        up, down = self.ffn[0], self.ffn[2]
        z = up(u) if W1 is None else torch.matmul(u, W1.mT) + up.bias
        y = down(F.gelu(z, approximate="tanh")) if W2 is None else torch.matmul(F.gelu(z, approximate="tanh"), W2.mT) + down.bias
        if write and self.ttt is not None and self.ttt.cfg.order == "rw":
            W1, W2 = self.ttt(u, W1, W2, self.ffn[0], self.ffn[2], y)
        return x + y, k, v, W1, W2


class Toy(nn.Module):
    def __init__(self, cfg, layers=4, dim=128, heads=4, ffn_dim=512, window=1):
        super().__init__()
        self.window = window
        self.slot = nn.Parameter(torch.randn(5, dim) * 0.1)     # 杯子位置（像视频里的空间位置）
        self.reveal = nn.Embedding(2, dim)                       # chunk 0：这个位置有没有球
        self.swap = nn.Embedding(6, dim)                         # 交换 chunk：这个位置要换到哪（0-4），5 = 不动
        self.query = nn.Parameter(torch.randn(dim) * 0.1)        # 最后一个 chunk
        self.blocks = nn.ModuleList([Block(dim, heads, ffn_dim) for _ in range(layers)])
        for i in cfg.layers:
            self.blocks[i].ttt = FastWriter(dim, cfg)
        self.head = nn.Linear(dim, 1)

    def chunk(self, x, st, write):
        # st = (每层窗口里的 K 列表, V 列表, W1 列表, W2 列表)
        ks, vs, W1s, W2s = st
        nk, nv, n1, n2 = [], [], list(W1s), list(W2s)
        for i, blk in enumerate(self.blocks):
            ck = torch.cat(ks[i], 2) if ks[i] else None
            cv = torch.cat(vs[i], 2) if vs[i] else None
            x, k, v, w1, w2 = blk(x, ck, cv, W1s[i], W2s[i], write)
            nk.append(k), nv.append(v)
            n1[i], n2[i] = w1, w2
        if write:
            # window=k>0：只看前 k 个 chunk；0：全看（注意力的上限对照）；-1：不看任何过去的 chunk（只能靠 fast weights）
            keep = (lambda l: l[len(l) - self.window:]) if self.window > 0 else (lambda l: l) if self.window == 0 else (lambda l: [])
            st = ([keep(o + [k]) for o, k in zip(ks, nk)], [keep(o + [v]) for o, v in zip(vs, nv)], n1, n2)
        return x, st

    def forward(self, start, swaps):
        B, N, _ = swaps.shape
        L = len(self.blocks)
        st = ([[] for _ in range(L)], [[] for _ in range(L)], [None] * L, [None] * L)
        slots = torch.arange(5, device=start.device)
        x = self.slot[None] + self.reveal((slots[None] == start[:, None]).long())             # chunk 0
        _, st = self.chunk(x, st, True)
        for k in range(N):
            a, b = swaps[:, k, :1], swaps[:, k, 1:]
            dest = torch.where(slots[None] == a, b, torch.where(slots[None] == b, a, torch.full_like(a, 5)))
            _, st = self.chunk(self.slot[None] + self.swap(dest), st, True)
        x, _ = self.chunk((self.slot + self.query)[None].expand(B, -1, -1), st, False)       # 只读
        return self.head(x).squeeze(-1)                                                        # [B, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--ttt-layers", default="all")
    ap.add_argument("--ttt-params", default="both")
    ap.add_argument("--ttt-loss", default="dot")
    ap.add_argument("--ttt-value", default="self")
    ap.add_argument("--ttt-order", default="wr")
    ap.add_argument("--ttt-lr", type=float, default=1e-2)
    ap.add_argument("--ttt-norm", type=int, default=1)
    ap.add_argument("--ttt-heads", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=5)
    ap.add_argument("--curriculum", type=int, default=1)  # 1：每步 N 从 0..n_train 均匀抽（只训 N=5 的话连全注意力都学不动）
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    L = args.layers
    layers = () if args.ttt_layers == "none" else tuple(range(L)) if args.ttt_layers == "all" else \
        tuple(range(L - int(args.ttt_layers[4:]), L)) if args.ttt_layers.startswith("last") else \
        tuple(int(i) for i in args.ttt_layers.split(","))
    cfg = TTTConfig(layers=layers, params=args.ttt_params, loss=args.ttt_loss, value=args.ttt_value,
                    order=args.ttt_order, lr=args.ttt_lr, norm=bool(args.ttt_norm), heads=args.ttt_heads)
    model = Toy(cfg, layers=L, window=args.window).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    name = args.name or f"L{L}_w{args.window}_{args.ttt_loss}_{args.ttt_params}_{args.ttt_layers}_{args.ttt_value}_{args.ttt_order}"
    t0 = time.time()
    hist = []
    for step in range(1, args.steps + 1):
        n = int(torch.randint(0, args.n_train + 1, ())) if args.curriculum else args.n_train  # 课程：每步 N ~ U{0..n_train}
        start, swaps, pos = episodes(args.batch, n, "cuda")
        loss = F.cross_entropy(model(start, swaps).float(), pos)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append(loss.item())
        if step % 1000 == 0:
            with torch.no_grad():
                st5, sw5, p5 = episodes(1024, args.n_train, "cuda")
                a5 = (model(st5, sw5).argmax(-1) == p5).float().mean().item()
            print(f"{name} step {step} loss {sum(hist[-200:]) / 200:.3f} acc@N{args.n_train} {a5:.2f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    acc = {}
    with torch.no_grad():
        for n in (1, 3, 5, 10, 20, 30):
            start, swaps, pos = episodes(2048, n, "cuda")
            acc[n] = (model(start, swaps).argmax(-1) == pos).float().mean().item()
    print(f"RESULT {name}: " + " ".join(f"N{n} {a:.2f}" for n, a in acc.items()), flush=True)
    out = Path("/project/sds-rise/ethan/Finetune/world/auto_research/toy")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}_s{args.seed}.json").write_text(json.dumps({"args": vars(args), "acc": acc}))


if __name__ == "__main__":
    main()
