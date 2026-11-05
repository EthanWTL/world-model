# 旋转方块的 base（从头训）：diffusion forcing（每个 chunk 独立 σ，一半样本带干净前缀）、整段 chunk 因果注意力（全看得见）、
# NorMuon（论文协议 2e-4）。数据 = Dataset/CubeMemory/train（gap ~ U{0, 2, 4, 6}，6–12 个 chunk）。
# 每 --eval-every 步 rank 0 用 EMA 评一次（逐 chunk + KV cache 全留，tf + free，gap 0 / 2 / 6）
# 输出 Finetune/world/auto_research/visual/<run>/：last.pt（续训）、ema_XXXXXX.pt（train.py 的 --base）、log.jsonl
#   CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node 2 train_base.py --run base
#   sbatch -J base train_base.slurm --run base
import argparse
import copy
import datetime
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from data import train_loader
from normuon import NorMuon
from wm import WanMem

OUT = "/project/sds-rise/ethan/Finetune/world/auto_research/visual"


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="base")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--size", default="200m")
    ap.add_argument("--gaps", type=int, nargs="+", default=[0, 2, 4, 6])
    ap.add_argument("--batch", type=int, default=16)  # 全局 batch，每张卡 batch // 卡数
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--p-clean", type=float, default=0.5)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--eval-num", type=int, default=32)
    ap.add_argument("--eval-gaps", type=int, nargs="+", default=[0, 2, 6])
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--keep-every", type=int, default=2500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", type=int, default=1)
    return ap.parse_args()


class Seq(nn.Module):
    # DDP 包的是 forward：这里的 forward = 整段前向
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, a):
        return self.model.forward_seq(x, t, a)


def diffusion_forcing_loss(net, x0, a, p_clean=0.5, shift=3.0):
    # 和 world_model/diffusion.py 一样：每个 chunk 一个 σ（shift 3）；以 p_clean 的概率把开头 1..K-1 个 chunk 设成干净（σ=0，不算 loss）
    B, K = a.shape
    u = torch.rand(B, K, device=x0.device)
    sigma = shift * u / (1 + (shift - 1) * u)
    length = torch.randint(1, K, (B,), device=x0.device) * (torch.rand(B, device=x0.device) < p_clean)
    clean = torch.arange(K, device=x0.device)[None] < length[:, None]
    sigma = sigma.masked_fill(clean, 0.0)
    s = sigma.repeat_interleave(5, 1)[:, None, :, None, None]
    noise = torch.randn_like(x0)
    pred = net((1 - s) * x0 + s * noise, sigma * 1000, a)
    err = (pred.float() - (noise - x0)).pow(2).mean((1, 3, 4)).view(B, K, 5).mean(-1)  # [B, K]
    keep = ~clean
    return (err * keep).sum() / keep.sum()


@torch.no_grad()
def update_ema(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1 - decay)


def main():
    args = get_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if world > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=1))  # rank 0 评估的时候别的卡在等
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out) / args.run

    model = WanMem(args.size, max_ctx=None, grad_ckpt=bool(args.grad_ckpt)).cuda()
    mats = [m.weight for m in model.modules() if isinstance(m, nn.Linear)]
    ids = {id(w) for w in mats}
    opt = NorMuon(mats, [p for p in model.parameters() if id(p) not in ids], lr=args.lr)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    step, tokens, gpu_h = 0, 0, 0.0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cuda")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, tokens, gpu_h = ck["step"], ck.get("tokens", 0), ck.get("gpu_hours", 0.0)
    net = DDP(Seq(model), device_ids=[torch.cuda.current_device()]) if world > 1 else Seq(model)

    def loader():  # 数据可能还在边渲染边编码：每次评估后重建一次，新编好的 episode 也进来
        return train_loader(args.batch // world, args.gaps, num_workers=args.workers,
                            seed=args.seed + 1000 * rank + step, seed_gap=args.seed + step)

    batches = loader()

    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        from evaluate import Evaluator
        evaluator = Evaluator(tuple(args.eval_gaps), num=args.eval_num, steps=20, modes=("tf", "free"))
        log = open(out / "log.jsonl", "a")
        use_wandb = False
        if args.wandb:
            try:
                import wandb
                id_file = out / "wandb_id"
                if not id_file.exists():
                    id_file.write_text(wandb.util.generate_id())
                wandb.init(project="world-auto-visual", name=args.run, id=id_file.read_text(), resume="allow",
                           dir=str(out), config=vars(args))
                use_wandb = True
            except Exception as e:
                print(f"wandb off: {e}", flush=True)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"run {args.run} | {args.size} {n_par / 1e6:.1f}M | {world} GPU x batch {args.batch // world} | "
              f"gaps {args.gaps} | start step {step}", flush=True)

    def record(d):
        log.write(json.dumps(d) + "\n")
        log.flush()
        if use_wandb:
            wandb.log({k: v for k, v in d.items() if k != "step"}, step=d["step"])

    def evaluate():
        if rank == 0:
            res = evaluator.run(ema)
            print(f"step {step} eval | " + " ".join(f"{k} {v:.3f}" for k, v in res.items()
                                                   if k.endswith(("cell_acc", "_psnr"))), flush=True)
            record({"step": step, **{f"eval/{k}": v for k, v in res.items()}})
        if world > 1:
            dist.barrier()

    if step == 0 and args.eval_every:
        evaluate()
    hist, t0 = [], time.time()
    while step < args.steps:
        b = next(batches)
        x0, a = b["latent"].cuda(non_blocking=True), b["actions"].cuda(non_blocking=True)
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = diffusion_forcing_loss(net, x0, a, args.p_clean)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        update_ema(ema, model, args.ema)
        step += 1
        tokens += x0.shape[0] * world * a.shape[1] * 960
        hist.append((loss.item(), gn.item(), a.shape[1]))
        if step % args.log_every == 0:
            dt = (time.time() - t0) / len(hist)
            gpu_h += dt * len(hist) * world / 3600
            if rank == 0:
                d = {"step": step, "loss": sum(h[0] for h in hist) / len(hist),
                     "grad_norm": sum(h[1] for h in hist) / len(hist), "s_per_step": dt,
                     "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "tokens": tokens, "gpu_hours": gpu_h}
                for K in sorted({h[2] for h in hist}):
                    hk = [h[0] for h in hist if h[2] == K]
                    d[f"loss_K{K}"] = sum(hk) / len(hk)
                print(f"step {step} | loss {d['loss']:.4f} | gn {d['grad_norm']:.2f} | {dt:.2f} s/step | "
                      f"{d['mem_gb']:.0f} GB | {tokens / 1e9:.2f}B tok", flush=True)
                record(d)
            hist, t0 = [], time.time()
        if rank == 0 and step % args.save_every == 0:
            tmp = out / "last.tmp"
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                        "tokens": tokens, "gpu_hours": gpu_h, "args": vars(args)}, tmp)
            os.replace(tmp, out / "last.pt")
        if rank == 0 and step % args.keep_every == 0:
            torch.save({"ema": ema.state_dict(), "step": step, "tokens": tokens, "gpu_hours": gpu_h, "args": vars(args)},
                       out / f"ema_{step:06d}.pt")
        if args.eval_every and step % args.eval_every == 0:
            evaluate()
            batches = loader()
            t0 = time.time()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
