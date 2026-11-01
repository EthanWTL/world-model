# 评估：最后一次揭开时球在不在对的杯子下面
#   tf  ：除了最后一个 chunk 都 commit GT，只生成最后一个（没有生成误差累积，只看记忆）
#   free：论文的协议，只给 2 个 GT chunk（揭开、盖上），后面全部自己生成
# 对 tf 还算一个"软"指标：σ=1（纯噪声）时一步预测的 x0 = 只靠上下文/记忆猜的平均画面，
#   解码后 5 个杯子位置各有多少红色 -> p_true（对的杯子占的比例，0.2 = 瞎猜，1 = 确定）
#   python evaluate.py --ckpt /project/sds-rise/ethan/Finetune/world/auto_research/<run>/ema_006000.pt
#   python evaluate.py --base .../ema_160000.pt --max-ctx 7      （基础模型，不微调）
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import default_collate

from data import ShellGameDataset
from detect import ball_slot
from wm import TTTConfig, WanTTT, commit, load_base

BASE = "/project/sds-rise/ethan/Finetune/world/shellgame_normuon/ema_160000.pt"


def sigma_schedule(steps=50, shift=3.0, device="cuda"):
    u = torch.linspace(1, 0, steps + 1, device=device)
    return shift * u / (1 + (shift - 1) * u)


class VAE:
    def __init__(self):
        from diffusers import AutoencoderKLWan
        self.vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae",
                                                    torch_dtype=torch.float32).cuda().eval()
        self.mean = torch.tensor(self.vae.config.latents_mean, device="cuda").view(1, 16, 1, 1, 1)
        self.std = torch.tensor(self.vae.config.latents_std, device="cuda").view(1, 16, 1, 1, 1)

    @torch.no_grad()
    def decode(self, z):
        # z: [B, 16, T, 24, 32] -> [B, 4T-3, 192, 256, 3] float [0, 1]
        x = self.vae.decode(z.float() * self.std + self.mean).sample
        return ((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 4, 1)


def slot_mass(frames, num_cups=5, tau=0.2):
    # frames: [B, T, H, W, 3] in [0, 1]（取最后 4 帧，杯子已经抬起来了）-> [B, 5] 每个位置的红色量；桌面 r-max(g,b)≈0.1–0.15，球 >0.25
    f = frames[:, -4:]
    red = (f[..., 0] - f[..., 1:].amax(-1) - tau).clamp(min=0).sum(1)  # [B, H, W]
    return torch.stack([r.sum((-1, -2)) for r in red.chunk(num_cups, dim=-1)], -1)


@torch.no_grad()
def sample_chunk(model, st, a, shape, steps, gen):
    sig = sigma_schedule(steps, device="cuda")
    x = torch.randn(shape, device="cuda", generator=gen)
    for i in range(steps):
        v, _ = model(x, sig[i].expand(shape[0]) * 1000, a, st)
        x = x + (sig[i + 1] - sig[i]) * v.float()
    return x


def commit_chunk(model, x, a, st, gen):
    sig_in = noise_in = None
    if model.ttt.loss == "e2e" and model.ttt.layers:  # e2e 的 inner loss 要噪声；固定 generator，结果可复现
        B = x.shape[0]
        s = model.ttt.e2e_sigma
        sig_in = torch.full((B,), s, device="cuda") if s >= 0 else torch.rand(B, device="cuda", generator=gen)
        sig_in = 3 * sig_in / (1 + 2 * sig_in) if s < 0 else sig_in
        noise_in = torch.randn(x.shape, device="cuda", generator=gen)
    return commit(model, x, a, st, sig_in, noise_in)


@torch.no_grad()
def final_chunk(model, x0, a, n_ctx, steps, seed):
    # 前 n_ctx 个 chunk 用 GT，之后一直生成到最后；返回 (最后一个 chunk 的采样结果, σ=1 时一步预测的 x0)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    B, K = a.shape
    shape = (B, 16, 5, x0.shape[3], x0.shape[4])
    st = model.init_state()
    for c in range(K - 1):
        x = x0[:, :, 5 * c:5 * c + 5] if c < n_ctx else sample_chunk(model, st, a[:, c], shape, steps, gen)
        st = commit_chunk(model, x, a[:, c], st, gen)
    noise = torch.randn(shape, device="cuda", generator=gen)
    v, _ = model(noise, torch.full((B,), 1000.0, device="cuda"), a[:, -1], st)
    x0_hat = noise - v.float()  # σ=1：x_t = 噪声，x0 = x_t - σ·v
    return sample_chunk(model, st, a[:, -1], shape, steps, gen), x0_hat


def make_split(s):
    # "currN"：课程数据里交换 N 次的 episode（是训练集里的，只用来看早期有没有学到）；其余就是 Dataset/ShellGame 下的 split
    if s.startswith("curr"):
        ds = ShellGameDataset("/project/sds-rise/ethan/Dataset/ShellGameCurr", "train")
        ds.eps = [ep for ep in ds.eps if len(ep["actions"]) - 3 == int(s[4:])]
        return ds
    return ShellGameDataset(split=s)


class Evaluator:
    def __init__(self, splits=("test/n05", "test/n10", "test/n20"), num=32, batch=16, steps=20, modes=("tf",), seed=0):
        self.vae = VAE()
        self.data = {s: make_split(s) for s in splits}
        self.num, self.batch, self.steps, self.modes, self.seed = num, batch, steps, modes, seed

    @torch.no_grad()
    def run(self, model):
        was = model.training
        model.eval()
        res = {}
        for split, ds in self.data.items():
            name = split.split("/")[-1]
            for mode in self.modes:
                hit, hit1, ptrue, mse1 = [], [], [], []
                for i in range(0, min(self.num, len(ds)), self.batch):
                    b = default_collate([ds[j] for j in range(i, min(i + self.batch, self.num, len(ds)))])
                    x0, a, gt = b["latent"].cuda(), b["actions"].cuda(), b["ball"][:, -1]
                    n_ctx = a.shape[1] - 1 if mode == "tf" else 2
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        x, x0_hat = final_chunk(model, x0, a, n_ctx, self.steps, self.seed + i)
                    hit += [ball_slot(f[-1].cpu().numpy()) == g for f, g in zip(self.vae.decode(x), gt.tolist())]
                    m = slot_mass(self.vae.decode(x0_hat)).cpu()
                    p = torch.where(m.sum(1, keepdim=True) > 0, m / m.sum(1, keepdim=True).clamp(min=1e-8), 0.2)
                    ptrue += p[torch.arange(len(gt)), gt].tolist()
                    hit1 += (p.argmax(1) == gt).tolist()
                    mse1 += (x0_hat - x0[:, :, -5:]).pow(2).mean((1, 2, 3, 4)).tolist()  # 连续指标：σ=1 一步预测和 GT 差多少
                res[f"{name}_{mode}_acc"] = float(np.mean(hit))
                if mode == "tf":
                    res[f"{name}_tf_p"] = float(np.mean(ptrue))
                    res[f"{name}_tf_acc1"] = float(np.mean(hit1))
                    res[f"{name}_tf_mse1"] = float(np.mean(mse1))
        model.train(was)
        return res


def build_from_ckpt(path, key=None):
    ck = torch.load(path, map_location="cpu")
    args = ck["args"]
    model = WanTTT(ttt=ttt_config(args), max_ctx=args["max_ctx"]).cuda()
    sd = ck[key] if key else (ck["ema"] if "ema" in ck else ck["model"])
    model.load_state_dict(sd)
    return model, args, ck.get("step")


def parse_layers(s):
    if s in ("none", ""):
        return ()
    if s == "all":
        return tuple(range(12))
    if s.startswith("last"):
        return tuple(range(12 - int(s[4:]), 12))
    if s.startswith("first"):
        return tuple(range(int(s[5:])))
    return tuple(int(i) for i in s.split(","))


def ttt_config(args):
    return TTTConfig(layers=parse_layers(args["ttt_layers"]), params=args["ttt_params"], loss=args["ttt_loss"],
                     value=args["ttt_value"], order=args["ttt_order"], lr=args["ttt_lr"], norm=bool(args["ttt_norm"]),
                     v_init=args["ttt_v_init"], e2e_sigma=args.get("e2e_sigma", -1.0), heads=args.get("ttt_heads", 1),
                     fast_frac=args.get("ttt_fast_frac", 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)  # 微调出来的 checkpoint（里面存了 args）
    ap.add_argument("--key", default=None)   # last.pt 里选 "ema" 还是 "model"（默认 ema）
    ap.add_argument("--base", default=BASE)  # 不给 --ckpt 时：直接评基础模型
    ap.add_argument("--max-ctx", type=int, default=7)
    ap.add_argument("--splits", nargs="+", default=["test/n05", "test/n10", "test/n20", "test/n30"])
    ap.add_argument("--modes", nargs="+", default=["tf", "free"])
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    if args.ckpt:
        model, _, _ = build_from_ckpt(args.ckpt, args.key)
        out = Path(args.out or Path(args.ckpt).parent / "eval" / f"{Path(args.ckpt).stem}.json")
    else:
        model = load_base(WanTTT(max_ctx=args.max_ctx), args.base).cuda()
        out = Path(args.out or f"/project/sds-rise/ethan/Finetune/world/auto_research/base_eval/"
                               f"{Path(args.base).stem}_ctx{args.max_ctx}.json")
    ev = Evaluator(args.splits, args.num, args.batch, args.steps, tuple(args.modes))
    res = ev.run(model)
    for k, v in res.items():
        print(f"{k}: {v:.3f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "res": res}, indent=2))


if __name__ == "__main__":
    main()
