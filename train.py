import argparse
import copy
import os

import torch.nn as nn

from normuon import NorMuon

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import ShellGameDataset, infinite_loader      # Step 5 用
from diffusion import diffusion_forcing_loss            # Step 4 用
from model import WanModel

import time
from pathlib import Path

import wandb

def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/project/sds-rise/ethan/Dataset/ShellGame")
    ap.add_argument("--out", default="/project/sds-rise/ethan/Finetune/world/shellgame")
    ap.add_argument("--batch", type=int, default=16)        # 全局 batch，每张卡分 batch // 卡数
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--window", type=int, default=None)     # None：8 个 chunk 都能看；1：只看前一个 chunk
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=5000)    # 存 last.pt，续训用
    ap.add_argument("--keep-every", type=int, default=50000)   # 另外留一份 EMA 快照，eval 用
    ap.add_argument("--opt", default="adamw", choices=["adamw", "normuon"])
    return ap.parse_args(argv)

def setup(seed):
    # torchrun 会设好 RANK / WORLD_SIZE / LOCAL_RANK；直接 python train.py 就是单卡
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if world > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(seed + rank)  # 每张卡抽到的噪声不一样
    return rank, world

def build(args, world):
    model = WanModel(causal=True, window=args.window).cuda()
    if world > 1:
        model = DDP(model, device_ids=[torch.cuda.current_device()])  # 构造时会把 rank 0 的参数广播到所有卡
    ema = copy.deepcopy(getattr(model, "module", model)).requires_grad_(False)  # 参数的滑动平均，eval 时用它
    if args.opt == "normuon":
        mats = [m.weight for m in model.modules() if isinstance(m, nn.Linear)]  # 77 个 Linear 的权重矩阵
        ids = {id(w) for w in mats}
        others = [p for p in model.parameters() if id(p) not in ids]            # 其余 117 个：bias、norm、modulation、embedding、Conv3d
        opt = NorMuon(mats, others, lr=args.lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    return model, ema, opt

@torch.no_grad()
def update_ema(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1 - decay)  # e = decay * e + (1 - decay) * p

def train_step(model, ema, opt, batch, step, args):
    x0 = batch['latent'].cuda(non_blocking=True)
    a = batch['actions'].cuda(non_blocking=True)

    for g in opt.param_groups:
        g['lr'] = args.lr * min(1.0, (step+ 1) / args.warmup)

    with torch.autocast('cuda', dtype = torch.bfloat16):
        loss = diffusion_forcing_loss(model, x0, a) #return one loss value
    opt.zero_grad(set_to_none=True)
    loss.backward() #bcalculata gradient
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  #norm the gradient
    opt.step() #update
    update_ema(ema, model, args.ema)
    return loss.item(), grad_norm.item()

def save(path, model, ema, opt, step):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": getattr(model, "module", model).state_dict(), "ema": ema.state_dict(),
                "opt": opt.state_dict(), "step": step}, tmp)
    os.replace(tmp, path)  # 先写临时文件再改名：存到一半被杀掉，旧的 last.pt 也还是完好的


def main():
    args = get_args()
    rank, world = setup(args.seed)
    model, ema, opt = build(args, world)
    out, step = Path(args.out), 0
    if (out / "last.pt").exists():  # 续训：从上次存的地方接着跑
        ck = torch.load(out / "last.pt", map_location="cuda")
        getattr(model, "module", model).load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
    ds = ShellGameDataset(args.root, "train")
    batches = infinite_loader(ds, args.batch // world, num_workers=4, rank=rank, world_size=world, seed=args.seed + step)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        id_file = out / "wandb_id"  # run id 存在 out 里：续训读同一个；删掉文件夹重来就生成新的
        if not id_file.exists():
            id_file.write_text(wandb.util.generate_id())
        wandb.init(project="world", name=out.name, id=id_file.read_text(), resume="allow", dir=str(out), config=vars(args))
        print(f"{world} GPU x batch {args.batch // world}, {len(ds)} episodes, start at step {step}", flush=True)
    losses, t0 = [], time.time()
    while step < args.steps:
        loss, gn = train_step(model, ema, opt, next(batches), step, args)
        losses.append(loss)
        step += 1
        if step % args.log_every == 0:
            if rank == 0:
                loss, dt = sum(losses) / len(losses), (time.time() - t0) / len(losses)
                print(f"step {step} | loss {loss:.4f} | grad_norm {gn:.2f} | {dt:.2f} s/step", flush=True)
                wandb.log({"loss": loss, "grad_norm": gn, "lr": opt.param_groups[0]["lr"], "s_per_step": dt}, step=step)
            losses, t0 = [], time.time()
        if rank == 0 and step % args.save_every == 0:
            save(out / "last.pt", model, ema, opt, step)
        if rank == 0 and step % args.keep_every == 0:
            torch.save(ema.state_dict(), out / f"ema_{step:06d}.pt")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()