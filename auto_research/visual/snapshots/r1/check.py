# 正确性检查（改了 wm.py 之后重跑）：
#   [1] 逐 chunk 前向（KV cache 全留，前面的 chunk 干净）== 整段 chunk 因果前向
#   [2] 零初始化的记忆 == 没有记忆（各种配置，严格为 0）
#   [3] DeltaMem 手写的一步 == autograd 对 inner loss 求梯度（delta / dot，raw / norm key）
#   [4] BPTT：外层 loss 的梯度能传到记忆的目标 T 和步长参数（零初始化时也不为 0）
#   python check.py --gpu 0
import argparse

import torch

from wm import DeltaMem, MemConfig, WanMem, commit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--size", default="200m")
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    B, K = 2, 3
    base = WanMem(args.size, max_ctx=None).cuda().eval()
    for p in base.parameters():  # head 是零初始化的，随机一下才看得出差别
        p.data.add_(torch.randn_like(p) * 0.02)
    x = torch.randn(B, 16, 5 * K, 24, 32, device="cuda")
    a = torch.randint(0, 3, (B, K), device="cuda")
    t_last = torch.rand(B, device="cuda") * 1000

    with torch.no_grad():
        t = torch.zeros(B, K, device="cuda")
        t[:, -1] = t_last
        seq = base.forward_seq(x, t, a)[:, :, -5:]
        st = base.init_state()
        for c in range(K - 1):
            st = commit(base, x[:, :, 5 * c:5 * c + 5], a[:, c], st)
        one, _ = base(x[:, :, -5:], t_last, a[:, -1], st)
    print(f"[1] per-chunk vs forward_seq: max diff {(one - seq).abs().max().item():.2e} (scale {seq.abs().max().item():.2f})")

    sd = base.state_dict()
    for cfg in [MemConfig(proj="q", layers=tuple(range(12))), MemConfig(proj="v", layers=tuple(range(12))),
                MemConfig(proj="qkv", layers=(0, 5, 11), fast_heads=16, blocks=1),
                MemConfig(proj="kv", layers=tuple(range(12)), key="norm", lr=0.5),
                MemConfig(ffn=True, layers=tuple(range(12))), MemConfig(proj="v", ffn=True, layers=(3, 7))]:
        m = WanMem(args.size, mem=cfg, max_ctx=0).cuda().eval()
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected and all(".amem." in k or ".fmem." in k for k in missing)
        ref = WanMem(args.size, max_ctx=0).cuda().eval()
        ref.load_state_dict(sd)
        with torch.no_grad():
            outs = []
            for model in (m, ref):
                st = model.init_state()
                for c in range(K - 1):
                    st = commit(model, x[:, :, 5 * c:5 * c + 5], a[:, c], st)
                outs.append(model(x[:, :, -5:], t_last, a[:, -1], st)[0])
        print(f"[2] zero-init {cfg.proj or '-'} ffn={cfg.ffn} key={cfg.key}: diff {(outs[0] - outs[1]).abs().max().item():.1e}")

    for loss in ("delta", "dot"):
        for key in ("raw", "norm"):
            cfg = MemConfig(loss=loss, key=key, renorm=False)
            dm = DeltaMem(64, 32, 4, cfg).cuda()
            S = torch.randn(B, 32, 64, device="cuda") * 0.1
            xx, tt = torch.randn(B, 10, 64, device="cuda"), torch.randn(B, 10, 32, device="cuda")
            step = torch.rand(B, 10, 1, device="cuda")
            new = dm.write(S, xx, tt, step)
            Sg = S.clone().requires_grad_(True)
            k = dm.key(xx)
            pred = torch.matmul(k, Sg.mT)
            inner = (0.5 * (pred - tt).pow(2).sum(-1) if loss == "delta" else -(pred * tt).sum(-1)) * step[..., 0]
            g, = torch.autograd.grad(inner.mean(1).sum(), Sg)
            print(f"[3] {loss}/{key}: hand vs autograd {(new - (S - g * dm.mask)).abs().max().item():.1e}")

    m = WanMem(args.size, mem=MemConfig(proj="qv", layers=tuple(range(12))), max_ctx=0, grad_ckpt=True).cuda().train()
    m.load_state_dict(sd, strict=False)
    st = m.init_state()
    for c in range(K - 1):
        st = commit(m, x[:, :, 5 * c:5 * c + 5], a[:, c], st)
    out, _ = m(x[:, :, -5:], t_last, a[:, -1], st)
    out.pow(2).mean().backward()
    blk = m.blocks[6].amem
    print(f"[4] grad |T_q| {blk.t['q'].weight.grad.norm().item():.2e} |T_v| {blk.t['v'].weight.grad.norm().item():.2e} "
          f"|step| {blk.step.weight.grad.norm().item():.2e}")


if __name__ == "__main__":
    main()
