# 记忆的 TTT 微调：从旋转方块的 base（train_base.py 的 EMA）出发，max_ctx 0（不看任何过去的 chunk = 关掉历史 KV cache、
# 没有重复注入的首帧 / 历史图像），过去的视觉信息只能走记忆（原生权重上的 ΔW）。
# teacher forcing 把一个 episode 逐 chunk 展开，和推理一模一样：
#   chunk c：denoise 带噪的 chunk c（读记忆，算 loss；chunk 0 不算）-> commit 干净的 chunk c（写记忆）
# 外层 loss 穿过所有 commit 反传（BPTT），目标 t / 步长 / slow weights 学的是"写什么、写多少"
# 预算按累计处理的 latent token 记（另记 steps 和 GPU 小时）；训完自动用 EMA 做完整评估（所有 gap，tf + free）
#   python train.py --run v_h4 --mem-proj v --gpu 0
#   python train.py --run ctrl                       （不开记忆：max_ctx 0 下的"没有跨时间记忆"对照）
import argparse
import copy
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn as nn

from data import train_loader
from evaluate import OUT, Evaluator
from normuon import NorMuon
from wm import WanMem, commit, load_base, mem_config


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--base", default=f"{OUT}/base/ema_020000.pt")
    ap.add_argument("--base-gate", type=float, default=0.5)  # base 自己（KV 全留）在 gap 0 的 tf cell_acc 不到这个就不训（base 坏了别白跑）
    ap.add_argument("--size", default="200m")
    ap.add_argument("--max-ctx", type=int, default=0)
    # 记忆（消融接口）
    ap.add_argument("--mem-proj", default="")              # 注意力的哪些投影："q" / "k" / "v" / "qk" / "qkv" …；"" = 不开
    ap.add_argument("--mem-ffn", type=int, default=0)      # 1：FFN down 也做记忆
    ap.add_argument("--mem-layers", default="all")         # all / last6 / first6 / every2 / 0,4,8
    ap.add_argument("--mem-fast-heads", type=int, default=4)  # 16 个 head 里后几个做 fast（slots），其余兜底
    ap.add_argument("--mem-blocks", type=int, default=4)   # fast 部分切几块（块对角）
    ap.add_argument("--mem-ffn-frac", type=float, default=0.25)
    ap.add_argument("--mem-ffn-blocks", type=int, default=8)
    ap.add_argument("--mem-loss", default="delta", choices=["delta", "dot"])
    ap.add_argument("--mem-key", default="raw", choices=["raw", "norm"])
    ap.add_argument("--mem-lr", type=float, default=1e-3)  # raw：η 初值；norm：β 初值（0–2）
    ap.add_argument("--mem-renorm", type=int, default=1)
    ap.add_argument("--mem-lr-lr", type=float, default=1e-3)  # 步长参数自己的 AdamW 学习率
    # 优化
    ap.add_argument("--gaps", type=int, nargs="+", default=[0, 2, 4, 6])  # 训练 gap 的混合分布（均匀）
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    # 记录 / 评估
    ap.add_argument("--eval-every", type=int, default=1000)  # 快评：原始权重，tf，32 个，20 步
    ap.add_argument("--eval-gaps", type=int, nargs="+", default=[0, 2, 6, 12])
    ap.add_argument("--eval-num", type=int, default=32)
    ap.add_argument("--final-eval", type=int, default=1)     # 训完：EMA，所有 gap，tf + free，64 个，50 步，存视频
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


def unroll(model, x0, a):
    # 返回 (每个样本每个 chunk 的 loss [B, K-1]（chunk 1..K-1），展开完的 state)
    B, K = a.shape
    st, losses = model.init_state(), []
    for c in range(K):
        xc = x0[:, :, 5 * c:5 * c + 5]
        if c >= 1:  # chunk 0 推理时也是给定的，不算 loss
            sig = sample_sigma(B)
            noise = torch.randn_like(xc)
            s = sig[:, None, None, None, None]
            pred, _ = model((1 - s) * xc + s * noise, sig * 1000, a[:, c], st)
            losses.append((pred.float() - (noise - xc)).pow(2).mean(dim=(1, 2, 3, 4)))
        if c < K - 1:  # 最后一个 chunk 后面没东西了，不用 commit
            st = commit(model, xc, a[:, c], st)
    return torch.stack(losses, 1), st


@torch.no_grad()
def mem_stats(model, st):
    # 每层每个记忆：|ΔW| / |对应的 slow 行|（batch 平均）
    out = {}
    for i, blk in enumerate(model.blocks):
        mats = []
        if blk.amem is not None:
            mats += [(p, getattr(blk.self_attn, p).weight[-blk.amem.nf:]) for p in blk.amem.cfg.proj]
        if blk.fmem is not None:
            mats.append(("ffn", blk.ffn[2].weight[:, -blk.fmem.nf:]))
        for name, ref in mats:
            S = st.mem[i].get(name)
            if S is not None:
                out[f"dW_{name}_{i}"] = (S.float().norm(dim=(1, 2)) / ref.float().norm()).mean().item()
    return out


def build(args):
    model = WanMem(args.size, mem_config(vars(args)), max_ctx=args.max_ctx, grad_ckpt=bool(args.grad_ckpt))
    model = load_base(model, args.base).cuda().train()
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    # Linear 的权重矩阵走 NorMuon；步长参数（*.step.*）单独一组 AdamW，lr 大一些（在 softplus / sigmoid 里，挪得慢）；其余 AdamW
    step_params = [p for n, p in model.named_parameters() if ".step." in n]
    sid = {id(p) for p in step_params}
    mats = [m.weight for m in model.modules() if isinstance(m, nn.Linear) and id(m.weight) not in sid]
    ids = {id(w) for w in mats} | sid
    opt = NorMuon(mats, [p for p in model.parameters() if id(p) not in ids], lr=args.lr)
    if step_params:
        opt.add_param_group(dict(params=step_params, muon=False, lr=args.mem_lr_lr, weight_decay=0.0))
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    return model, ema, opt


@torch.no_grad()
def update_ema(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1 - decay)


def main():
    args = get_args()
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out) / args.run
    blog = Path(args.base).parent / "log.jsonl"
    if args.base_gate > 0 and blog.exists():
        bstep = torch.load(args.base, map_location="cpu").get("step", 0)
        ev = [r for r in map(json.loads, open(blog)) if "eval/g00_tf_cell_acc" in r and r["step"] <= bstep]
        if not ev or ev[-1]["eval/g00_tf_cell_acc"] < args.base_gate:
            raise SystemExit(f"base gate: {args.base} g00 tf cell_acc {ev[-1]['eval/g00_tf_cell_acc'] if ev else None} "
                             f"< {args.base_gate}")
    out.mkdir(parents=True, exist_ok=True)
    model, ema, opt = build(args)
    step, tokens, gpu_h = 0, 0, 0.0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cuda")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, tokens, gpu_h = ck["step"], ck["tokens"], ck["gpu_hours"]
    n_new = sum(p.numel() for n, p in model.named_parameters() if ".amem." in n or ".fmem." in n)
    n_w, n_s = model.mem_numel()
    info = {"mem_new_params": n_new, "mem_writable_per_sample": n_w, "mem_state_per_sample": n_s,
            "mem_state_mb_fp32": n_s * 4 / 2 ** 20, "gpu": torch.cuda.get_device_name()}
    print(f"run {args.run} | {model.mem_cfg} | max_ctx {args.max_ctx} | new params {n_new / 1e6:.2f}M | "
          f"fast weights / sample: writable {n_w / 1e6:.2f}M, state {n_s / 1e6:.2f}M ({n_s * 4 / 2 ** 20:.0f} MB fp32) | "
          f"start step {step}", flush=True)
    (out / "info.json").write_text(json.dumps({"args": vars(args), **info}, indent=1))
    batches = train_loader(args.batch, args.gaps, num_workers=args.workers, seed=args.seed + step)
    evaluator = Evaluator(tuple(args.eval_gaps), num=args.eval_num, steps=20, modes=("tf",))
    log = open(out / "log.jsonl", "a")
    use_wandb = False
    if args.wandb:
        try:
            import wandb
            id_file = out / "wandb_id"
            if not id_file.exists():
                id_file.write_text(wandb.util.generate_id())
            wandb.init(project="world-auto-visual", name=args.run, id=id_file.read_text(), resume="allow", dir=str(out),
                       config={**vars(args), **info})
            use_wandb = True
        except Exception as e:  # 没网 / 没登录：只写 log.jsonl
            print(f"wandb off: {e}", flush=True)

    def record(d):
        log.write(json.dumps(d) + "\n")
        log.flush()
        if use_wandb:
            wandb.log({k: v for k, v in d.items() if k != "step"}, step=d["step"])

    def quick_eval():
        res = evaluator.run(model)
        print(f"step {step} eval | " + " ".join(f"{k} {v:.3f}" for k, v in res.items()
                                               if k.endswith(("tf_cell_acc", "tf1_cell_acc", "tf_psnr"))), flush=True)
        record({"step": step, "tokens": tokens, "gpu_hours": gpu_h, **{f"eval/{k}": v for k, v in res.items()}})

    if step == 0:
        quick_eval()
    hist, t0 = [], time.time()
    while step < args.steps:
        b = next(batches)
        x0, a = b["latent"].cuda(non_blocking=True), b["actions"].cuda(non_blocking=True)
        K = a.shape[1]
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * min(1.0, (step + 1) / args.warmup)
        opt.zero_grad(set_to_none=True)
        Ls = []
        for xa, aa in zip(x0.chunk(args.accum), a.chunk(args.accum)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                L, st = unroll(model, xa, aa)
                loss = L.mean()
            ms = mem_stats(model, st) if (step + 1) % args.log_every == 0 else {}
            del st
            (loss / args.accum).backward()
            Ls.append(L.detach())
        L = torch.cat(Ls)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        update_ema(ema, model, args.ema)
        step += 1
        tokens += x0.shape[0] * K * 960
        hist.append((L.mean().item(), L.mean(0).cpu(), gn.item(), K))
        if step % args.log_every == 0:
            dt = (time.time() - t0) / len(hist)
            gpu_h += dt * len(hist) / 3600
            # 按阶段分开看 loss：观察（chunk 1–3，新面进来，记忆帮不上）/ 离开 4 / 间隔 / 返回（最后一个，目标面回来）
            ph = {"obs": [], "leave": [], "gap": [], "ret": []}
            for _, lc, _, k in hist:
                ph["obs"].append(lc[0:3].mean().item())
                ph["leave"].append(lc[3].item())
                if k > 6:
                    ph["gap"].append(lc[4:k - 2].mean().item())
                ph["ret"].append(lc[k - 2].item())
            d = {"step": step, "loss": sum(h[0] for h in hist) / len(hist), "grad_norm": sum(h[2] for h in hist) / len(hist),
                 "s_per_step": dt, "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "tokens": tokens,
                 "gpu_hours": gpu_h, **{f"loss_{k}": sum(v) / len(v) for k, v in ph.items() if v}}
            for k in sorted({h[3] for h in hist}):
                d[f"loss_ret_K{k}"] = sum(h[1][k - 2].item() for h in hist if h[3] == k) / sum(h[3] == k for h in hist)
            for name in ("q", "k", "v", "ffn"):
                vals = [v for kk, v in ms.items() if kk.startswith(f"dW_{name}_")]
                if vals:
                    d[f"dW_{name}_mean"] = sum(vals) / len(vals)
            d.update(ms)
            print(f"step {step} | loss {d['loss']:.4f} | obs {d['loss_obs']:.4f} ret {d['loss_ret']:.4f} | "
                  f"gn {d['grad_norm']:.2f} | " + "".join(f"dW_{p} {d[f'dW_{p}_mean']:.2e} " for p in ("q", "k", "v", "ffn")
                                                         if f"dW_{p}_mean" in d)
                  + f"| {dt:.2f} s/step | {d['mem_gb']:.0f} GB", flush=True)
            record(d)
            hist, t0 = [], time.time()
        if step % args.eval_every == 0 or step == args.steps:
            quick_eval()
            t0 = time.time()
        if step % args.save_every == 0 or step == args.steps:
            tmp = out / "last.tmp"
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                        "tokens": tokens, "gpu_hours": gpu_h, "args": vars(args)}, tmp)
            os.replace(tmp, out / "last.pt")
        if step % args.keep_every == 0 or step == args.steps:
            torch.save({"ema": ema.state_dict(), "model": model.state_dict(), "step": step, "tokens": tokens,
                        "gpu_hours": gpu_h, "args": vars(args)}, out / f"ckpt_{step:06d}.pt")
    if args.final_eval and not (out / "eval" / f"ckpt_{args.steps:06d}_ema.json").exists():
        from evaluate import TEST_GAPS
        res = Evaluator(TEST_GAPS, num=64, steps=50, modes=("tf", "free"), videos=2,
                        video_dir=out / "eval" / f"ckpt_{args.steps:06d}_ema").run(ema, per_episode=True)
        (out / "eval").mkdir(exist_ok=True)
        (out / "eval" / f"ckpt_{args.steps:06d}_ema.json").write_text(json.dumps(
            {"step": args.steps, "tokens": tokens, "gpu_hours": gpu_h, "res": res[0], "per_episode": res[1]}, indent=1))
        print("final eval | " + " ".join(f"{k} {v:.3f}" for k, v in res[0].items() if k.endswith("free_cell_acc")),
              flush=True)


if __name__ == "__main__":
    main()
