# 多球基础模型：从单球 NorMuon 基础模型（shellgame_normuon/ema_200000）接着训，数据换成 Dataset/ShellGameMulti
# （k = 1..5 个球混在一起，N = 0..5 的课程）。训法和 world_model/train.py 一样：diffusion forcing（每个 chunk 独立噪声，
# 一半样本带干净前缀）、全因果注意力（≤ 8 个 chunk 全看得见）、NorMuon。模型和 loss 用 ref/ 里的快照（model.py / diffusion.py）。
# 每 --eval-every 步 rank 0 用 EMA 评一次每个 k 的 test/n05（tf + free，max_ctx 7 = 全部上下文）：所有球都对才算对。
#   CUDA_VISIBLE_DEVICES=1,6 torchrun --standalone --nproc_per_node 2 train_base.py --run mb_base
# 输出 Finetune/world/auto_research/<run>/：last.pt（续训）、ema_XXXXXX.pt（可以直接当 train.py 的 --base）、log.jsonl
import argparse
import copy
import datetime
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(str(Path(__file__).resolve().parent / "ref"))  # 放在最后：model / diffusion 只有 ref/ 里有，detect 还是用这里的
from diffusion import diffusion_forcing_loss  # noqa: E402
from model import WanModel  # noqa: E402

from data import multi_curriculum_loader  # noqa: E402
from evaluate import Evaluator  # noqa: E402
from normuon import NorMuon  # noqa: E402
from wm import WanTTT  # noqa: E402


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mb_base")
    ap.add_argument("--out", default="/project/sds-rise/ethan/Finetune/world/auto_research")
    ap.add_argument("--init", default="/project/sds-rise/ethan/Finetune/world/shellgame_normuon/ema_200000.pt")
    ap.add_argument("--balls", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--batch", type=int, default=16)  # 全局 batch，每张卡 batch // 卡数（和基础模型一样）
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--eval-num", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--keep-every", type=int, default=2500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", type=int, default=1)
    return ap.parse_args()


@torch.no_grad()
def update_ema(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1 - decay)


def save(path, model, ema, opt, step, args):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                "args": vars(args)}, tmp)
    os.replace(tmp, path)


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

    model = WanModel(causal=True, window=None).cuda()
    mats = [m.weight for m in model.modules() if isinstance(m, nn.Linear)]
    ids = {id(w) for w in mats}
    opt = NorMuon(mats, [p for p in model.parameters() if id(p) not in ids], lr=args.lr)
    ema = copy.deepcopy(model).requires_grad_(False)
    step = 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cuda")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
    else:  # 从单球基础模型的 EMA 出发，优化器从头来
        sd = torch.load(args.init, map_location="cpu")
        model.load_state_dict(sd.get("ema", sd))
        ema.load_state_dict(model.state_dict())
    ddp = DDP(model, device_ids=[torch.cuda.current_device()]) if world > 1 else model
    batches = multi_curriculum_loader(args.batch // world, args.balls, args.workers,
                                      seed=args.seed + 1000 * rank + step, seed_n=args.seed + step)

    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        evaluator = Evaluator(tuple(f"k{k}/test/n05" for k in args.balls), num=args.eval_num, steps=20, modes=("tf", "free"))
        evm = WanTTT(max_ctx=7).cuda()  # 逐 chunk + KV cache 的推理模型（和 train.py / evaluate.py 一样），评估时装 EMA
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
            except Exception as e:
                print(f"wandb off: {e}", flush=True)
        print(f"run {args.run} | {world} GPU x batch {args.batch // world} | balls {args.balls} | start step {step}", flush=True)

    def record(d):
        log.write(json.dumps(d) + "\n")
        log.flush()
        if use_wandb:
            wandb.log({k: v for k, v in d.items() if k != "step"}, step=d["step"])

    def evaluate():
        if rank == 0:
            missing, unexpected = evm.load_state_dict(ema.state_dict(), strict=False)
            assert not missing and not unexpected, (missing, unexpected)
            res = evaluator.run(evm)
            print(f"step {step} eval | " + " ".join(f"{k} {v:.3f}" for k, v in res.items() if "_acc" in k or "_ball" in k),
                  flush=True)
            record({"step": step, **{f"eval/{k}": v for k, v in res.items()}})
        if world > 1:
            dist.barrier()

    if step == 0:
        evaluate()
    hist, t0 = [], time.time()
    while step < args.steps:
        b = next(batches)
        x0, a = b["latent"].cuda(non_blocking=True), b["actions"].cuda(non_blocking=True)
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = diffusion_forcing_loss(ddp, x0, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        update_ema(ema, model, args.ema)
        step += 1
        hist.append((loss.item(), gn.item(), a.shape[1] - 3))
        if step % args.log_every == 0:
            if rank == 0:
                dt = (time.time() - t0) / len(hist)
                d = {"step": step, "loss": sum(h[0] for h in hist) / len(hist),
                     "grad_norm": sum(h[1] for h in hist) / len(hist), "s_per_step": dt,
                     "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
                for n in sorted({h[2] for h in hist}):
                    hn = [h[0] for h in hist if h[2] == n]
                    d[f"loss_n{n}"] = sum(hn) / len(hn)
                print(f"step {step} | loss {d['loss']:.4f} | gn {d['grad_norm']:.2f} | {dt:.2f} s/step | "
                      f"{d['mem_gb']:.0f} GB", flush=True)
                record(d)
            hist, t0 = [], time.time()
        if rank == 0 and step % args.save_every == 0:
            save(out / "last.pt", model, ema, opt, step, args)
        if rank == 0 and step % args.keep_every == 0:
            torch.save(ema.state_dict(), out / f"ema_{step:06d}.pt")
        if step % args.eval_every == 0:
            evaluate()
            t0 = time.time()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
