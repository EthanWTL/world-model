# 正确性检查（改了 wm.py 就跑一遍）：
#   1. 不开 TTT 的 WanTTT 逐 chunk rollout == ref/model.py 一次前向 8 个 chunk（fp32 下比）
#   2. 开 TTT 但 Wv 零初始化：输出和不开 TTT 完全一样
#   3. FastWriter 手写的梯度 == autograd（dot / mse，up / down）
#   4. 部分单元 fast + heads 的手写梯度 == autograd
#   5. 注意力 q/k/v（AttnWriter，前面的 head 兜底 + heads）的手写梯度 == autograd
#   6. key=proj（写入的 key = x + W_k·x）：FFN / 注意力的手写梯度 == autograd
#   python check.py --gpu 0
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "ref"))
from model import WanModel  # noqa: E402  ref/ 里的快照，不碰 world_model/ 下的原文件

from data import ShellGameDataset  # noqa: E402
from wm import FastWriter, TTTConfig, WanTTT, commit, load_base  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--base", default="/project/sds-rise/ethan/Finetune/world/shellgame_normuon/ema_160000.pt")
args = ap.parse_args()
torch.cuda.set_device(args.gpu)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

ds = ShellGameDataset("/project/sds-rise/ethan/Dataset/ShellGame", "test/n05")
x0 = torch.stack([ds[i]["latent"] for i in range(2)]).cuda()
a = torch.stack([ds[i]["actions"] for i in range(2)]).cuda()
B, K = a.shape

# ---- 1. 逐 chunk == 一次前向
ref = WanModel(causal=True).cuda().eval()
ref.load_state_dict(torch.load(args.base, map_location="cpu"))
sigma = torch.zeros(B, K, device="cuda")
sigma[:, -1] = 0.7
torch.manual_seed(0)
noise = torch.randn_like(x0[:, :, -5:])
xt = x0.clone()
xt[:, :, -5:] = 0.3 * x0[:, :, -5:] + 0.7 * noise
with torch.no_grad():
    want = ref(xt, sigma * 1000, a)[:, :, -5:]
    m = load_base(WanTTT(max_ctx=7), args.base).cuda().eval()  # 7 = 8 个 chunk 全在窗口里（也测窗口裁剪）
    st = m.init_state()
    for c in range(K - 1):
        st = commit(m, x0[:, :, 5 * c:5 * c + 5], a[:, c], st)
    got, _ = m(xt[:, :, -5:], sigma[:, -1] * 1000, a[:, -1], st)
print(f"[1] 逐 chunk vs 一次前向：max |diff| {(got - want).abs().max():.2e}，|want| {want.abs().mean():.3f}")

# ---- 2. Wv 零初始化 = 不开 TTT
for cfg in (TTTConfig(layers=tuple(range(12)), params="both", loss="dot"),
            TTTConfig(layers=tuple(range(12)), params="both", loss="mse", order="rw"),
            TTTConfig(layers=(9, 10, 11), params="both", loss="dot", value="next"),
            TTTConfig(layers=tuple(range(12)), params="both", loss="dot", value="read", order="rw", heads=8),
            TTTConfig(layers=tuple(range(12)), params="both", loss="mse", heads=8, fast_frac=0.5),
            TTTConfig(layers=(9, 10, 11), params="both", loss="e2e", heads=8, fast_frac=0.25),
            TTTConfig(layers=tuple(range(12)), params="none", attn="qkv", loss="mse", heads=4, fast_frac=0.25),
            TTTConfig(layers=tuple(range(12)), params="none", attn="v", loss="dot", order="rw"),
            TTTConfig(layers=tuple(range(12)), params="down", attn="qkv", loss="mse", heads=4, fast_frac=0.25),
            TTTConfig(layers=(9, 10, 11), params="none", attn="qkv", loss="e2e", heads=4, fast_frac=0.25),
            TTTConfig(layers=tuple(range(12)), params="down", loss="mse", heads=8, fast_frac=0.25, key="proj"),
            TTTConfig(layers=tuple(range(12)), params="none", attn="q", loss="mse", heads=4, fast_frac=0.25, key="proj")):
    with torch.no_grad():
        mt = load_base(WanTTT(ttt=cfg, max_ctx=1), args.base).cuda().eval()
        m1 = load_base(WanTTT(max_ctx=1), args.base).cuda().eval()
        outs = []
        for mm in (mt, m1):
            st = mm.init_state()
            for c in range(K - 1):
                xc = x0[:, :, 5 * c:5 * c + 5]
                if mm.ttt.loss == "e2e" and mm.ttt.layers:  # e2e 的 inner loss 要噪声
                    st = commit(mm, xc, a[:, c], st, torch.full((B,), 0.5, device="cuda"), torch.randn_like(xc))
                else:
                    st = commit(mm, xc, a[:, c], st)
            outs.append(mm(xt[:, :, -5:], sigma[:, -1] * 1000, a[:, -1], st)[0])
    print(f"[2] {cfg.loss}/{cfg.params}/attn={cfg.attn or '-'}/{cfg.order}/{cfg.value}/h{cfg.heads}/f{cfg.fast_frac} 零初始化 vs 无 TTT："
          f"max |diff| {(outs[0] - outs[1]).abs().max():.2e}" + ("（e2e 一开始就有更新，不该是 0）" if cfg.loss == "e2e" else ""))

# ---- 3. 手写梯度 vs autograd
torch.manual_seed(0)
D, Fd, T = 64, 256, 40
up, down = torch.nn.Linear(D, Fd).cuda(), torch.nn.Linear(Fd, D).cuda()
u = torch.randn(3, T, D, device="cuda")
for loss in ("dot", "mse"):
    cfg = TTTConfig(params="both", loss=loss, norm=False, lr=0.1)
    w = FastWriter(D, cfg).cuda()
    torch.nn.init.normal_(w.v.weight, std=0.3)
    torch.nn.init.normal_(w.lr.weight, std=0.05)
    W1 = (up.weight + 0.05 * torch.randn(3, Fd, D, device="cuda")).requires_grad_(True)
    W2 = (down.weight + 0.05 * torch.randn(3, D, Fd, device="cuda")).requires_grad_(True)
    with torch.no_grad():
        n1, n2 = w(u, W1, W2, up, down)
    v = w.v(u)
    eta = torch.nn.functional.softplus(w.lr(u))
    y = torch.matmul(torch.nn.functional.gelu(torch.matmul(u, W1.mT) + up.bias, approximate="tanh"), W2.mT)
    if loss == "dot":
        per_tok = -(y * v).sum(-1)
    else:
        y0 = torch.nn.functional.linear(torch.nn.functional.gelu(up(u), approximate="tanh"), down.weight)
        per_tok = 0.5 * (y - y0 - v.detach()).pow(2).sum(-1)
    # 每个 token 的 lr 不一样：W1 的梯度按 eta[...,0] 加权，W2 按 eta[...,1]
    g1 = torch.autograd.grad((per_tok * eta[..., 0].detach()).sum() / T, W1, retain_graph=True)[0]
    g2 = torch.autograd.grad((per_tok * eta[..., 1].detach()).sum() / T, W2)[0]
    e1 = ((W1 - g1) - n1).abs().max() / g1.abs().max()
    e2 = ((W2 - g2) - n2).abs().max() / g2.abs().max()
    print(f"[3] {loss}: W1 相对误差 {e1:.2e}，W2 相对误差 {e2:.2e}")

# ---- 4. 部分单元 fast（兜底）+ heads：手写梯度 vs autograd（loss 用 ffn_apply 写，slow 单元不该有梯度）
from wm import ffn_apply, head_mask  # noqa: E402
ns, H = 96, 4
Ff = Fd - ns
for loss in ("dot", "mse"):
    cfg = TTTConfig(params="both", loss=loss, norm=False, lr=0.1, heads=H)
    w = FastWriter(D, cfg).cuda()
    torch.nn.init.normal_(w.v.weight, std=0.3)
    torch.nn.init.normal_(w.lr.weight, std=0.05)
    W1 = (up.weight[ns:] + 0.05 * torch.randn(3, Ff, D, device="cuda")).requires_grad_(True)
    W2 = (down.weight[:, ns:] + 0.05 * torch.randn(3, D, Ff, device="cuda")).requires_grad_(True)
    with torch.no_grad():
        n1, n2 = w(u, W1, W2, up, down, ns)
    v = w.v(u).detach()
    eta = torch.nn.functional.softplus(w.lr(u)).detach()
    y = ffn_apply(up, down, u, W1, W2, ns) - down.bias
    if loss == "dot":
        per_tok = -(y * v).sum(-1)
    else:
        y0 = torch.nn.functional.linear(torch.nn.functional.gelu(up(u), approximate="tanh"), down.weight)
        per_tok = 0.5 * (y - y0 - v).pow(2).sum(-1)
    g1 = torch.autograd.grad((per_tok * eta[..., 0]).sum() / T, W1, retain_graph=True)[0] * head_mask(Ff, D, H, "cuda")
    g2 = torch.autograd.grad((per_tok * eta[..., 1]).sum() / T, W2)[0] * head_mask(D, Ff, H, "cuda")
    e1 = ((W1 - g1) - n1).abs().max() / g1.abs().max()
    e2 = ((W2 - g2) - n2).abs().max() / g2.abs().max()
    print(f"[4] {loss} 部分 fast（ns={ns}）+ {H} heads：W1 相对误差 {e1:.2e}，W2 相对误差 {e2:.2e}")

# ---- 5. 注意力 q/k/v 的 fast 输出通道（前面的 head 兜底）+ heads：手写梯度 vs autograd
from wm import AttnWriter, SelfAttention, proj_apply  # noqa: E402
attn = SelfAttention(D, 4).cuda()
na, H = 32, 2
R = D - na
for loss in ("dot", "mse"):
    cfg = TTTConfig(params="none", attn="qkv", loss=loss, norm=False, lr=0.1, heads=H)
    w = AttnWriter(D, R, cfg).cuda()
    for m in w.t.values():
        torch.nn.init.normal_(m.weight, std=0.3)
    torch.nn.init.normal_(w.lr.weight, std=0.05)
    Wa = {p: (getattr(attn, p).weight[na:] + 0.05 * torch.randn(3, R, D, device="cuda")).requires_grad_(True) for p in "qkv"}
    with torch.no_grad():
        new = w(u, Wa, attn, na)
    eta = torch.nn.functional.softplus(w.lr(u)).detach()
    errs = []
    for j, p in enumerate("qkv"):
        lin = getattr(attn, p)
        y = proj_apply(lin, u, Wa[p], na)[..., na:] - lin.bias[na:]  # fast 通道的输出（不含 bias）
        t = w.t[p](u).detach()
        if loss == "dot":
            per_tok = -(y * t).sum(-1)
        else:
            per_tok = 0.5 * (y - torch.nn.functional.linear(u, lin.weight[na:]) - t).pow(2).sum(-1)
        g = torch.autograd.grad((per_tok * eta[..., j]).sum() / T, Wa[p])[0] * head_mask(R, D, H, "cuda")
        errs.append(((Wa[p] - g) - new[p]).abs().max() / g.abs().max())
    print(f"[5] attn {loss}（na={na}）+ {H} heads：q/k/v 相对误差 " + " ".join(f"{e:.2e}" for e in errs))

# ---- 6. key=proj：写入的 key 换成 x + W_k·x（v、lr 还是从 x 来）
for which in ("ffn", "attn"):
    if which == "ffn":
        cfg = TTTConfig(params="down", loss="mse", norm=False, lr=0.1, heads=H, key="proj")
        w = FastWriter(D, cfg).cuda()
        torch.nn.init.normal_(w.v.weight, std=0.3)
    else:
        cfg = TTTConfig(params="none", attn="q", loss="mse", norm=False, lr=0.1, heads=H, key="proj")
        w = AttnWriter(D, R, cfg).cuda()
        torch.nn.init.normal_(w.t["q"].weight, std=0.3)
    torch.nn.init.normal_(w.k.weight, std=0.2)
    torch.nn.init.normal_(w.lr.weight, std=0.05)
    uk = (u + w.k(u)).detach()
    eta = torch.nn.functional.softplus(w.lr(u)).detach()
    if which == "ffn":
        W2 = (down.weight[:, ns:] + 0.05 * torch.randn(3, D, Ff, device="cuda")).requires_grad_(True)
        with torch.no_grad():
            _, n2 = w(u, None, W2, up, down, ns)
        h = torch.nn.functional.gelu(up(uk), approximate="tanh")[..., ns:]
        per_tok = 0.5 * (torch.matmul(h, (W2 - down.weight[:, ns:]).mT) - w.v(u).detach()).pow(2).sum(-1)
        g = torch.autograd.grad((per_tok * eta[..., 1]).sum() / T, W2)[0] * head_mask(D, Ff, H, "cuda")
        err = ((W2 - g) - n2).abs().max() / g.abs().max()
    else:
        Wq = (attn.q.weight[na:] + 0.05 * torch.randn(3, R, D, device="cuda")).requires_grad_(True)
        with torch.no_grad():
            nq = w(u, {"q": Wq}, attn, na)["q"]
        per_tok = 0.5 * (torch.matmul(uk, (Wq - attn.q.weight[na:]).mT) - w.t["q"](u).detach()).pow(2).sum(-1)
        g = torch.autograd.grad((per_tok * eta[..., 0]).sum() / T, Wq)[0] * head_mask(R, D, H, "cuda")
        err = ((Wq - g) - nq).abs().max() / g.abs().max()
    print(f"[6] key=proj {which}：相对误差 {err:.2e}")
