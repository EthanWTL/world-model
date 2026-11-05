# TTT 模型的评估：最后一次揭开时，球在不在对的杯子下面（Wan2.1 VAE 只解码最后一个 chunk，ball_slot 认最后一帧）
#   tf  ：除了最后一个 chunk 都 commit GT，只生成最后一个（没有生成误差的累积，只看记忆）
#   free：论文的协议，只给 2 个 GT chunk（揭开、盖上），后面全部自己生成、自己 commit
#   mse1：σ = 1（纯噪声）时一步预测的 x0 和 GT 的 latent MSE。连续指标；fast v 在训练长度外崩掉时它会暴涨（auto 的坑 18）
#   python ttt_eval.py --ckpt /project/sds-rise/ethan/Finetune/world/ttt/<run>/ckpt_012000.pt
#   python ttt_eval.py --splits n05 --modes tf --num 16     （不给 --ckpt：评 base，窗口 7，n05 tf 应该 ≈ 1.0）
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # Wan2.1 VAE 已经在 ~/.cache/huggingface 里

import numpy as np
import torch
from torch.utils.data import default_collate

from data import ShellGameDataset
from diffusion import sigma_schedule
from eval import VAE
from shellgame.detect import ball_slot
from ttt_model import TTTModel, load_base

BASE = "/project/sds-rise/ethan/Finetune/world/shellgame_normuon/ema_160000.pt"
ROOT = "/project/sds-rise/ethan/Dataset/ShellGame"
CURR = "/project/sds-rise/ethan/Dataset/ShellGameCurr"


def parse_layers(s, n=12):
    # "all" / "0,4,8"
    return tuple(range(n)) if s == "all" else tuple(int(i) for i in s.split(","))


def build_model(args):
    # 训练和评估共用：从参数（ttt_train.py 的 vars(args)，或者 checkpoint 里存的 args）建一个 TTTModel
    return TTTModel(proj=args["mem_proj"], layers=parse_layers(args["mem_layers"]), fast_heads=args["mem_fast_heads"],
                    heads=args["mem_heads"], loss=args["mem_loss"], eta0=args["mem_eta0"], renorm=bool(args["mem_renorm"]),
                    max_ctx=args["max_ctx"], grad_ckpt=bool(args["grad_ckpt"]))


def make_split(name):
    # "currN"：课程数据里交换 N 次的（训练集里的，只用来看早期学没学到）；"n05" .. "n30"：ShellGame/test 下的
    if name.startswith("curr"):
        ds = ShellGameDataset(CURR, "train")
        ds.eps = [ep for ep in ds.eps if len(ep["actions"]) - 3 == int(name[4:])]
        return ds
    return ShellGameDataset(ROOT, f"test/{name}")


@torch.no_grad()
def sample_chunk(model, st, a, shape, steps, gen):
    # 从纯噪声开始 Euler 走 steps 步，每一步只读 state
    sig = sigma_schedule(steps, device="cuda")
    x = torch.randn(shape, device="cuda", generator=gen)
    for i in range(steps):
        v, _ = model(x, sig[i].expand(shape[0]) * 1000, a, st)
        x = x + (sig[i + 1] - sig[i]) * v.float()
    return x


@torch.no_grad()
def last_chunk(model, x0, a, n_ctx, steps, seed):
    # 前 n_ctx 个 chunk 用 GT，之后的自己生成；除了最后一个，每个 chunk 都 commit 进 state
    # 返回 (最后一个 chunk 的采样结果, σ = 1 时一步预测的 x0)，都是 [B, 16, 5, 24, 32]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    B, K = a.shape
    shape = (B, 16, 5, x0.shape[3], x0.shape[4])
    st = model.init_state(B)
    for c in range(K - 1):
        x = x0[:, :, 5 * c:5 * c + 5] if c < n_ctx else sample_chunk(model, st, a[:, c], shape, steps, gen)
        st = model.commit(x, a[:, c], st)
    noise = torch.randn(shape, device="cuda", generator=gen)
    v, _ = model(noise, torch.full((B,), 1000.0, device="cuda"), a[:, -1], st)
    x0_hat = noise - v.float()  # σ = 1：x_t 就是噪声，x0 = x_t - σ·v
    return sample_chunk(model, st, a[:, -1], shape, steps, gen), x0_hat


class Evaluator:
    def __init__(self, splits=("curr3", "n05", "n10", "n20"), num=32, steps=20, modes=("tf",), batch=16, seed=0):
        self.vae = VAE()
        self.data = {s: make_split(s) for s in splits}
        self.num, self.steps, self.modes, self.batch, self.seed = num, steps, modes, batch, seed

    @torch.no_grad()
    def run(self, model):
        was = model.training
        model.eval()
        res = {}
        for name, ds in self.data.items():
            n = min(self.num, len(ds))
            for mode in self.modes:
                hit, mse1 = [], []
                for i in range(0, n, self.batch):
                    b = default_collate([ds[j] for j in range(i, min(i + self.batch, n))])
                    x0, a, gt = b["latent"].cuda(), b["actions"].cuda(), b["ball"][:, -1].tolist()
                    n_ctx = a.shape[1] - 1 if mode == "tf" else 2
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        x, x0_hat = last_chunk(model, x0, a, n_ctx, self.steps, self.seed + i)
                    frames = self.vae.decode(x.float())  # [B, 17, 192, 256, 3] uint8
                    hit += [ball_slot(f[-1]) == g for f, g in zip(frames, gt)]
                    mse1 += (x0_hat - x0[:, :, -5:]).pow(2).mean((1, 2, 3, 4)).tolist()
                res[f"{name}_{mode}_acc"] = float(np.mean(hit))
                res[f"{name}_{mode}_mse1"] = float(np.mean(mse1))
        model.train(was)
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)   # ttt_train.py 存的 ckpt_XXXXXX.pt / last.pt（里面有 args）；不给 = 评 base
    ap.add_argument("--key", default="ema")   # "ema" / "model"（原始权重）
    ap.add_argument("--max-ctx", type=int, default=None)  # 不给：ckpt 用训练时的，base 用 7
    ap.add_argument("--splits", nargs="+", default=["n05", "n10", "n20", "n30"])
    ap.add_argument("--modes", nargs="+", default=["tf", "free"])
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        model = build_model(ck["args"])
        model.load_state_dict(ck[args.key])
        out = Path(args.out or Path(args.ckpt).parent / "eval" / f"{Path(args.ckpt).stem}_{args.key}.json")
    else:
        model = load_base(TTTModel(proj="", max_ctx=7), BASE)
        out = Path(args.out or "/project/sds-rise/ethan/Finetune/world/ttt/base_eval/ema_160000.json")
    if args.max_ctx is not None:
        model.max_ctx = args.max_ctx
    res = Evaluator(args.splits, args.num, args.steps, args.modes, args.batch).run(model.cuda())
    for k, v in res.items():
        print(f"{k}: {v:.4f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "max_ctx": model.max_ctx, "res": res}, indent=2))


if __name__ == "__main__":
    main()
