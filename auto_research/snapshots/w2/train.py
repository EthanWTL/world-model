# TTT 微调：从基础模型（shellgame_normuon 的 EMA）出发，窗口 max_ctx（默认 1 = 每个 chunk 只看前一个 chunk）+ FFN fast weights。
# teacher forcing 展开一个 episode 的 8 个 chunk（和推理时的 rollout 完全一样）：
#   chunk c：denoise 带噪的 chunk c（读 state，算 loss）-> commit 干净的 chunk c（写 state：K/V 窗口 + fast weights）
# 外层 loss 穿过所有 commit 反传（BPTT），所以 Wv / lr / slow weights 学的是"写什么、怎么写"
#   python train.py --run dot_both_all --ttt-layers all --ttt-params both --ttt-loss dot
#   python train.py --run ctrl_swa1                       （不开 TTT：SWA1 对照）
import argparse
import copy
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn as nn

from data import ShellGameDataset, infinite_loader
from evaluate import BASE, Evaluator, parse_layers, ttt_config
from normuon import NorMuon
from wm import BallProbe, FastWriter, WanTTT, commit, load_base


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="/project/sds-rise/ethan/Finetune/world/auto_research")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--max-ctx", type=int, default=1)
    ap.add_argument("--ttt-layers", default="none")   # none / all / last3 / first3 / 0,4,8
    ap.add_argument("--ttt-params", default="both", choices=["up", "down", "both"])
    ap.add_argument("--ttt-loss", default="dot", choices=["dot", "mse", "e2e"])
    ap.add_argument("--ttt-value", default="self", choices=["self", "next"])
    ap.add_argument("--ttt-order", default="wr", choices=["wr", "rw"])
    ap.add_argument("--ttt-lr", type=float, default=1e-3)
    ap.add_argument("--ttt-norm", type=int, default=1)
    ap.add_argument("--ttt-v-init", type=float, default=0.0)
    ap.add_argument("--e2e-sigma", type=float, default=-1.0)
    ap.add_argument("--ttt-heads", type=int, default=1)
    ap.add_argument("--probe", type=int, default=1)       # 诊断用的 probe（特征 detach，不影响模型）
    ap.add_argument("--aux", type=float, default=0.0)     # >0：probe 的 CE 乘这个权重反传进模型（用了球的标签，只当上限实验）
    ap.add_argument("--ttt-lr-lr", type=float, default=1e-3)  # inner lr 参数自己的（外层）学习率
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--last-weight", type=float, default=1.0)  # 最后一个 chunk（揭开）的 loss 权重
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-num", type=int, default=32)
    ap.add_argument("--eval-at-start", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--keep-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--wandb", type=int, default=1)
    return ap.parse_args(argv)


def sample_sigma(B, shift=3.0):
    u = torch.rand(B, device="cuda")
    return shift * u / (1 + (shift - 1) * u)


def unroll(model, x0, a, probe=None, aux=0.0):
    # 返回 (每个样本每个 chunk 的 loss [B, K-1]（chunk 1..K-1），最后的 state，probe 的 logits [B, K-2, 5]（chunk 1..K-2）)
    # probe：读 denoise 时最后一层的特征猜"这个 chunk 结束时球在哪"；aux=0 时特征 detach（纯诊断，不影响模型）
    B, K = a.shape
    st = model.init_state()
    losses, logits = [], []
    for c in range(K):
        xc = x0[:, :, 5 * c:5 * c + 5]
        if c >= 1:  # chunk 0 没有上下文，推理时也是给定的，不算 loss
            sig = sample_sigma(B)
            noise = torch.randn_like(xc)
            s = sig[:, None, None, None, None]
            pred, _, hid = model((1 - s) * xc + s * noise, sig * 1000, a[:, c], st, hidden=True)
            losses.append((pred.float() - (noise - xc)).pow(2).mean(dim=(1, 2, 3, 4)))
            if probe is not None and c < K - 1:  # 最后一个 chunk 球露出来了，不算
                logits.append(probe(hid if aux > 0 else hid.detach()))
        if c < K - 1:  # 最后一个 chunk 后面没东西了，不用 commit
            sig_in = noise_in = None
            if model.ttt.loss == "e2e" and model.ttt.layers:
                sig_in = sample_sigma(B) if model.ttt.e2e_sigma < 0 else torch.full((B,), model.ttt.e2e_sigma, device="cuda")
                noise_in = torch.randn_like(xc)
            st = commit(model, xc, a[:, c], st, sig_in, noise_in)
    return torch.stack(losses, 1), st, (torch.stack(logits, 1) if logits else None)


@torch.no_grad()
def fw_stats(model, st):
    # 每个 TTT 层：commit 完 7 个 chunk 后 fast weight 离 slow weight 多远（相对 Frobenius 范数，batch 平均）
    out = {}
    for i in model.ttt.layers:
        blk = model.blocks[i]
        for name, W, ref in (("up", st.W1[i], blk.ffn[0].weight), ("down", st.W2[i], blk.ffn[2].weight)):
            if W is not None and W.dim() == 3:
                out[f"dW_{name}_{i}"] = ((W.float() - ref.float()).norm(dim=(1, 2)) / ref.float().norm()).mean().item()
    return out


def build(args):
    model = WanTTT(ttt=ttt_config(vars(args)), max_ctx=args.max_ctx, grad_ckpt=bool(args.grad_ckpt))
    model = load_base(model, args.base).cuda().train()
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    # 三组：Linear 的权重矩阵走 NorMuon；inner lr 的参数（FastWriter.lr、e2e_lr）单独一组 AdamW，lr 大一些
    # （它们在 softplus 里，AdamW 每步只挪 ~lr，用 1e-4 的话要几万步 inner lr 才变得动）；其余 AdamW
    lr_params = [p for n, p in model.named_parameters() if ".ttt.lr." in n or n == "e2e_lr"]
    lr_ids = {id(p) for p in lr_params}
    mats = [m.weight for m in model.modules() if isinstance(m, nn.Linear) and id(m.weight) not in lr_ids]
    ids = {id(w) for w in mats} | lr_ids
    others = [p for p in model.parameters() if id(p) not in ids]
    opt = NorMuon(mats, others, lr=args.lr)
    if lr_params:
        opt.add_param_group(dict(params=lr_params, muon=False, lr=args.ttt_lr_lr, weight_decay=0.0))
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    return model, ema, opt


@torch.no_grad()
def update_ema(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1 - decay)


def save(path, model, ema, opt, step, args, probe=None, popt=None):
    tmp = path.with_suffix(".tmp")
    extra = {"probe": probe.state_dict(), "popt": popt.state_dict()} if probe is not None else {}
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                "args": vars(args), **extra}, tmp)
    os.replace(tmp, path)


def main():
    args = get_args()
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out) / args.run
    out.mkdir(parents=True, exist_ok=True)
    model, ema, opt = build(args)
    probe = BallProbe(model.dim).cuda() if args.probe or args.aux > 0 else None
    popt = torch.optim.AdamW(probe.parameters(), lr=1e-3) if probe is not None else None
    step = 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cuda")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        if probe is not None and "probe" in ck:
            probe.load_state_dict(ck["probe"])
            popt.load_state_dict(ck["popt"])
        step = ck["step"]
    n_new = sum(p.numel() for n, p in model.named_parameters() if ".ttt." in n or n == "e2e_lr")
    print(f"run {args.run} | TTT {model.ttt} | max_ctx {args.max_ctx} | new params {n_new / 1e6:.2f}M | start step {step}",
          flush=True)
    batches = infinite_loader(ShellGameDataset(split="train"), args.batch, args.workers, seed=args.seed + step)
    evaluator = Evaluator(("test/n05", "test/n10", "test/n20"), num=args.eval_num, steps=20)
    log = open(out / "log.jsonl", "a")
    use_wandb = False
    if args.wandb:
        try:
            import wandb
            id_file = out / "wandb_id"
            if not id_file.exists():
                id_file.write_text(wandb.util.generate_id())
            wandb.init(project="world-auto", name=args.run, id=id_file.read_text(), resume="allow", dir=str(out),
                       config=vars(args))
            use_wandb = True
        except Exception as e:  # 没网 / 没登录：只写 log.jsonl
            print(f"wandb off: {e}", flush=True)

    def record(d):
        log.write(json.dumps(d) + "\n")
        log.flush()
        if use_wandb:
            wandb.log({k: v for k, v in d.items() if k != "step"}, step=d["step"])

    if step == 0 and args.eval_at_start:  # 微调前（= 基础模型换成 max_ctx 窗口）先评一次
        res = evaluator.run(model)
        print(f"step 0 eval | " + " ".join(f"{k} {v:.3f}" for k, v in res.items()), flush=True)
        record({"step": 0, **{f"eval/{k}": v for k, v in res.items()}})
    w = None
    hist, t0 = [], time.time()
    while step < args.steps:
        b = next(batches)
        x0, a = b["latent"].cuda(non_blocking=True), b["actions"].cuda(non_blocking=True)
        if w is None:
            w = torch.ones(a.shape[1] - 1, device="cuda")
            w[-1] = args.last_weight
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * min(1.0, (step + 1) / args.warmup)
        opt.zero_grad(set_to_none=True)
        if popt is not None:
            popt.zero_grad(set_to_none=True)
        Ls, hits, ces = [], [], []
        ball = b["ball"].cuda(non_blocking=True)
        for xa, aa, ba in zip(x0.chunk(args.accum), a.chunk(args.accum), ball.chunk(args.accum)):  # 显存不够时拆成几份累积梯度
            with torch.autocast("cuda", dtype=torch.bfloat16):
                L, st, lg = unroll(model, xa, aa, probe, args.aux)
                loss = (L * w).sum(1).mean() / w.sum()
            if lg is not None:
                tgt = ba[:, 1:lg.shape[1] + 1]                                                 # chunk 1..K-2 结束时球在哪
                ce = torch.nn.functional.cross_entropy(lg.float().flatten(0, 1), tgt.flatten())
                loss = loss + (args.aux if args.aux > 0 else 1.0) * ce  # aux=0 时 logits 是 detach 的特征算的，只训 probe
                hits.append((lg.argmax(-1) == tgt).float().mean(0).detach())
                ces.append(ce.detach())
            fw = fw_stats(model, st) if (step + 1) % args.log_every == 0 else {}
            del st  # 每个样本的 fast weights 很大，别留到下一步
            (loss / args.accum).backward()
            Ls.append(L.detach())
        L = torch.cat(Ls)
        loss = (L * w).sum(1).mean() / w.sum()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if popt is not None:
            popt.step()
        update_ema(ema, model, args.ema)
        step += 1
        hist.append((loss.item(), L.detach().mean(0).cpu(), gn.item(),
                     torch.stack(hits).mean(0).cpu() if hits else None, torch.stack(ces).mean().item() if ces else None))
        if step % args.log_every == 0:
            dt = (time.time() - t0) / len(hist)
            per_chunk = torch.stack([h[1] for h in hist]).mean(0)
            d = {"step": step, "loss": sum(h[0] for h in hist) / len(hist), "loss_last": per_chunk[-1].item(),
                 "grad_norm": sum(h[2] for h in hist) / len(hist), "s_per_step": dt,
                 "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            d.update({f"loss_c{c + 1}": v for c, v in enumerate(per_chunk.tolist())})
            if hist[0][3] is not None:  # probe：每个 chunk 能不能从特征里读出球的位置（0.2 = 读不出）
                acc = torch.stack([h[3] for h in hist]).mean(0)
                d.update({f"probe_acc_c{c + 1}": v for c, v in enumerate(acc.tolist())})
                d["probe_ce"] = sum(h[4] for h in hist) / len(hist)
            for name in ("up", "down"):
                vals = [v for k, v in fw.items() if k.startswith(f"dW_{name}_")]
                if vals:
                    d[f"dW_{name}_mean"] = sum(vals) / len(vals)
            d.update(fw)
            if model.ttt.loss == "e2e" and model.ttt.layers:
                d["e2e_lr_mean"] = torch.nn.functional.softplus(model.e2e_lr.detach().float())[list(model.ttt.layers)].mean().item()
            pa = " ".join(f"{d[k]:.2f}" for k in sorted(d) if k.startswith("probe_acc_c"))
            print(f"step {step} | loss {d['loss']:.4f} | last {d['loss_last']:.4f} | gn {d['grad_norm']:.2f} | "
                  f"dW up {d.get('dW_up_mean', 0):.2e} down {d.get('dW_down_mean', 0):.2e} | probe {pa} | "
                  f"{dt:.2f} s/step | {d['mem_gb']:.0f} GB", flush=True)
            record(d)
            hist, t0 = [], time.time()
        if step % args.eval_every == 0 or step == args.steps:
            res = evaluator.run(model)
            print(f"step {step} eval | " + " ".join(f"{k} {v:.3f}" for k, v in res.items()), flush=True)
            record({"step": step, **{f"eval/{k}": v for k, v in res.items()}})
            t0 = time.time()
        if step % args.save_every == 0 or step == args.steps:
            save(out / "last.pt", model, ema, opt, step, args, probe, popt)
        if step % args.keep_every == 0 or step == args.steps:
            torch.save({"ema": ema.state_dict(), "model": model.state_dict(), "step": step, "args": vars(args)},
                       out / f"ckpt_{step:06d}.pt")


if __name__ == "__main__":
    main()
