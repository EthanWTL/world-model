# 训练 TTT 记忆（注意力 q/k/v 投影上的 fast weight：memory.py + ttt_model.py），从 base 的 EMA 出发
# teacher forcing 把一个 episode 逐 chunk 展开，和推理时完全一样：
#   chunk c：denoise 加了噪声的 chunk c（读 state，算 loss；chunk 0 不算）-> commit 干净的 chunk c（写记忆）
# 外层 loss 穿过所有 commit 反传（BPTT），所以 target / eta / slow weights 学的是"写什么、写多少"
# 默认 = 现在最好的组合（auto 的 wh_q）：q、delta、后 4 个 head fast、分 4 块、12 层、max_ctx 0、课程、12000 步
# 训完自动用 EMA 做完整评估（n05/10/20/30，tf + free，64 个 episode，50 步）-> <out>/<run>/eval/
#   python ttt_train.py --run q_delta --gpu 0
#   python ttt_train.py --run k_delta --mem-proj k --gpu 1        （消融：换投影 / loss / head / 层，见下面的参数）
import argparse
import copy
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import ShellGameDataset, curriculum_loader, infinite_loader
from diffusion import add_noise, sample_sigma
from normuon import NorMuon
from ttt_eval import BASE, ROOT, Evaluator, build_model
from ttt_model import load_base


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="/project/sds-rise/ethan/Finetune/world/ttt")
    ap.add_argument("--base", default=BASE)
    # 记忆（消融接口）
    ap.add_argument("--mem-proj", default="q")                # 哪些投影开记忆："q" / "k" / "v" / "qk" / "qkv"；"" = 不开
    ap.add_argument("--mem-layers", default="all")            # "all" / "0,4,8"
    ap.add_argument("--mem-fast-heads", type=int, default=4)  # 16 个注意力 head 里后几个做 fast，其余兜底（16 = 没有兜底）
    ap.add_argument("--mem-heads", type=int, default=4)       # fast 部分切成几块（块对角；1 = 不分块）
    ap.add_argument("--mem-loss", default="delta", choices=["delta", "dot"])
    ap.add_argument("--mem-eta0", type=float, default=1e-3)   # inner lr η 的初值
    ap.add_argument("--mem-renorm", type=int, default=1)
    ap.add_argument("--max-ctx", type=int, default=0)         # 每个 chunk 还能看前面几个 chunk 的 K/V；训练记忆时 0
    ap.add_argument("--curriculum", type=int, default=1)      # 1：每步 N ~ U{0..5}；0：只用 N = 5
    # 优化
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=1e-4)         # NorMuon（矩阵）+ AdamW（其余）
    ap.add_argument("--eta-lr", type=float, default=1e-3)     # η 的参数（mems.*.eta）单独一组 AdamW
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    # 记录 / 评估
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)    # 快评：原始权重，tf，32 个 episode，20 步
    ap.add_argument("--eval-splits", nargs="+", default=["curr3", "n05", "n10", "n20"])
    ap.add_argument("--eval-num", type=int, default=32)
    ap.add_argument("--final-eval", type=int, default=1)      # 训完用 EMA 做完整评估
    ap.add_argument("--save-every", type=int, default=500)    # last.pt，续训用
    ap.add_argument("--keep-every", type=int, default=2000)   # ckpt_XXXXXX.pt（model + ema + args），评估用
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--wandb", type=int, default=1)
    return ap.parse_args(argv)


def unroll(model, x0, a):
    # x0: [B, 16, 5K, 24, 32] 干净的 latent；a: [B, K]
    # 返回 (每个样本每个 chunk 的 loss [B, K-1]（chunk 1..K-1），展开完的 state）
    B, K = a.shape
    st, losses = model.init_state(B), []
    for c in range(K):
        xc = x0[:, :, 5 * c:5 * c + 5]
        if c >= 1:  # chunk 0 推理时也是给定的，不算 loss
            sig = sample_sigma(B, 1, device=x0.device)               # [B, 1]，shift 3
            xt, target = add_noise(xc, sig)                          # target = noise - x0（速度）
            pred, _ = model(xt, sig[:, 0] * 1000, a[:, c], st)       # 读 state
            losses.append((pred.float() - target).pow(2).mean((1, 2, 3, 4)))
        if c < K - 1:  # 最后一个 chunk 后面没有了，不用 commit
            st = model.commit(xc, a[:, c], st)                       # 写 state
    return torch.stack(losses, 1), st


def build(args):
    model = load_base(build_model(vars(args)), args.base).cuda().train()
    ema = copy.deepcopy(model).requires_grad_(False)  # 参数的滑动平均，评估用它
    # 三组：Linear 的权重矩阵走 NorMuon（包括记忆的 target）；其余（bias、norm、modulation、embedding、Conv3d）走 AdamW
    # η 的参数（mems.*.eta）单独一组 AdamW：它们在 softplus 里，lr 1e-4 的话要几万步才推得动（auto 的坑 2）
    eta = [p for n, p in model.named_parameters() if ".eta." in n]
    eta_ids = {id(p) for p in eta}
    mats = [m.weight for m in model.modules() if isinstance(m, nn.Linear) and id(m.weight) not in eta_ids]
    ids = {id(p) for p in mats} | eta_ids
    others = [p for p in model.parameters() if id(p) not in ids]
    opt = NorMuon(mats, others, lr=args.lr)
    if eta:
        opt.add_param_group(dict(params=eta, muon=False, lr=args.eta_lr, weight_decay=0.0))
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]  # warmup 按各组自己的 lr 放大
    return model, ema, opt


@torch.no_grad()
def update_ema(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1 - decay)  # e = decay * e + (1 - decay) * p


@torch.no_grad()
def mem_stats(model, st):
    # 展开完以后每个投影的记忆有多大：|ΔW| / |W_fast|（batch 平均、层平均）；η 的 bias 对应的步长（层平均）
    d = {}
    for i, mems in enumerate(model.mems):
        for p, m in mems.items():
            Ws = getattr(model.blocks[i].self_attn, p).weight[-m.nf:]
            d.setdefault(f"dW_{p}", []).append((st.mem[i][p].norm(dim=(1, 2)) / Ws.norm()).mean().item())
            d.setdefault(f"eta_{p}", []).append(F.softplus(m.eta.bias).item())
    return {k: sum(v) / len(v) for k, v in d.items()}


def save(path, model, ema, opt, step, args):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                "args": vars(args)}, tmp)
    os.replace(tmp, path)  # 先写临时文件再改名：存到一半被杀掉，旧的 last.pt 也还是完好的


def main():
    args = get_args()
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out) / args.run
    out.mkdir(parents=True, exist_ok=True)
    model, ema, opt = build(args)
    step = 0
    if (out / "last.pt").exists():  # 续训：从上次存的地方接着跑
        ck = torch.load(out / "last.pt", map_location="cuda")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
    n_new = sum(p.numel() for n, p in model.named_parameters() if n.startswith("mems."))
    print(f"run {args.run} | proj '{args.mem_proj}' {args.mem_loss} | fast heads {args.mem_fast_heads} | blocks "
          f"{args.mem_heads} | max_ctx {args.max_ctx} | new params {n_new / 1e6:.2f}M | start step {step}", flush=True)
    if args.curriculum:
        batches = curriculum_loader(args.batch, args.workers, seed=args.seed + step)
    else:
        batches = infinite_loader(ShellGameDataset(ROOT, "train"), args.batch, args.workers, seed=args.seed + step)
    evaluator = Evaluator(tuple(args.eval_splits), num=args.eval_num, steps=20)
    log = open(out / "log.jsonl", "a")
    use_wandb = False
    if args.wandb:
        try:
            import wandb
            id_file = out / "wandb_id"  # run id 存在 out 里：续训读同一个
            if not id_file.exists():
                id_file.write_text(wandb.util.generate_id())
            wandb.init(project="world-ttt", name=args.run, id=id_file.read_text(), resume="allow", dir=str(out),
                       config=vars(args))
            use_wandb = True
        except Exception as e:  # 没网 / 没登录：只写 log.jsonl
            print(f"wandb off: {e}", flush=True)

    def record(d):
        log.write(json.dumps(d) + "\n")
        log.flush()
        if use_wandb:
            wandb.log({k: v for k, v in d.items() if k != "step"}, step=d["step"])

    hist, t0 = [], time.time()
    while step < args.steps:
        b = next(batches)
        x0, a = b["latent"].cuda(non_blocking=True), b["actions"].cuda(non_blocking=True)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * min(1.0, (step + 1) / args.warmup)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            L, st = unroll(model, x0, a)  # [B, K-1]
        loss = L.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        update_ema(ema, model, args.ema)
        step += 1
        ms = mem_stats(model, st) if step % args.log_every == 0 else {}
        del st  # 每个样本的 ΔW 和整段计算图，别留到下一步
        hist.append((loss.item(), L[:, -1].mean().item(), gn.item(), a.shape[1] - 3))  # 最后一个 chunk = 揭开，要靠记忆
        if step % args.log_every == 0:
            dt = (time.time() - t0) / len(hist)
            d = {"step": step, "loss": sum(h[0] for h in hist) / len(hist),
                 "loss_last": sum(h[1] for h in hist) / len(hist), "grad_norm": sum(h[2] for h in hist) / len(hist),
                 "lr": opt.param_groups[0]["lr"], "s_per_step": dt, "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                 **ms}
            for n in sorted({h[3] for h in hist}):  # 按交换次数分开看最后一个 chunk 的 loss
                hn = [h[1] for h in hist if h[3] == n]
                d[f"loss_last_n{n}"] = sum(hn) / len(hn)
            print(f"step {step} | loss {d['loss']:.4f} | last {d['loss_last']:.4f} | gn {d['grad_norm']:.3f} | "
                  + " ".join(f"{k} {v:.2e}" for k, v in ms.items()) + f" | {dt:.2f} s/step | {d['mem_gb']:.0f} GB",
                  flush=True)
            record(d)
            hist, t0 = [], time.time()
        if step % args.eval_every == 0 or step == args.steps:  # 快评用原始权重：EMA 跟得慢，看不出什么时候学会
            res = evaluator.run(model)
            print(f"step {step} eval | " + " ".join(f"{k} {v:.3f}" for k, v in res.items() if k.endswith("acc")),
                  flush=True)
            record({"step": step, **{f"eval/{k}": v for k, v in res.items()}})
            t0 = time.time()
        if step % args.save_every == 0 or step == args.steps:
            save(out / "last.pt", model, ema, opt, step, args)
        if step % args.keep_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "step": step, "args": vars(args)},
                       out / f"ckpt_{step:06d}.pt")
    if args.final_eval:  # 完整评估：EMA，n05/10/20/30，tf + free，64 个 episode，50 步
        res = Evaluator(("n05", "n10", "n20", "n30"), num=64, steps=50, modes=("tf", "free")).run(ema.cuda())
        print("final (EMA) | " + " ".join(f"{k} {v:.3f}" for k, v in res.items() if k.endswith("acc")), flush=True)
        (out / "eval").mkdir(exist_ok=True)
        (out / "eval" / f"ckpt_{step:06d}_ema.json").write_text(json.dumps({"step": step, "res": res}, indent=2))
        record({"step": step, **{f"final/{k}": v for k, v in res.items()}})


if __name__ == "__main__":
    main()
