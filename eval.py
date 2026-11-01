import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # Wan2.1 VAE 已经在 ~/.cache/huggingface 里，不用联网

import imageio
import numpy as np
import torch
from diffusers import AutoencoderKLWan
from torch.utils.data import default_collate

from data import ShellGameDataset
from model import WanModel
from rollout import rollout
from shellgame.detect import ball_slot

def load_model(path, key="ema", window=None):
    # last.pt 里同时有 "model" 和 "ema"；ema_XXXXXX.pt 里只有 EMA 的参数
    ck = torch.load(path, map_location="cpu")
    model = WanModel(causal=True, window=window).cuda().eval()
    model.load_state_dict(ck[key] if key in ck else ck)
    return model

class VAE:
    # Wan2.1 VAE 解码：把 encode_latents.py 存下来的归一化 latent 变回 RGB 视频（就是编码那一步反过来）
    def __init__(self):
        self.vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae",
                                                    torch_dtype=torch.float32).cuda().eval()
        self.mean = torch.tensor(self.vae.config.latents_mean, device="cuda").view(1, 16, 1, 1, 1)
        self.std = torch.tensor(self.vae.config.latents_std, device="cuda").view(1, 16, 1, 1, 1)

    @torch.no_grad()
    def decode(self, z):
        # z: [B, 16, T, 24, 32] -> [B, 4T-3, 192, 256, 3] uint8：第 1 个 latent 帧解出 1 帧，之后每个解出 4 帧
        x = self.vae.decode(z * self.std + self.mean).sample  # [B, 3, 4T-3, 192, 256]，范围 [-1, 1]
        return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 4, 1).cpu().numpy()


def save_mp4(path, frames, fps=16):
    # frames: [T, H, W, 3] uint8
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)

def evaluate(model, vae, ds, num, batch, steps, max_ctx, out, save, name):
    # 前 2 个 chunk（揭开、盖上）用 GT，后面全部生成；看最后一帧里球是不是在正确的杯子下面
    hits = []
    for i in range(0, min(num, len(ds)), batch):
        b = default_collate([ds[j] for j in range(i, min(i + batch, num, len(ds)))])
        x0, a = b["latent"].cuda(), b["actions"].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video = rollout(model, x0[:, :, :10], a, steps=steps, max_ctx=max_ctx)
        gen = vae.decode(video)  # [B, T, 192, 256, 3]
        for j, frames in enumerate(gen):
            hits.append(ball_slot(frames[-1]) == b["ball"][j, -1].item())
            if i + j < save:  # 存前几个：左 GT，右生成
                save_mp4(out / f"{name}_{i + j}.mp4", np.concatenate([vae.decode(x0[j:j + 1])[0], frames], axis=2))
    return sum(hits) / len(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--key", default="ema")  # last.pt 里选 "ema" 还是 "model"
    ap.add_argument("--splits", nargs="+", default=[f"test/n{n:02d}" for n in range(5, 31, 5)])
    ap.add_argument("--num", type=int, default=64)  # 每个 split 评多少个 episode
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--max-ctx", type=int, default=7)  # 当前 chunk + 前 7 个，和训练长度一样
    ap.add_argument("--save", type=int, default=4)  # 每个 split 存几个 mp4
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--root", default="/project/sds-rise/ethan/Dataset/ShellGame")
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    ckpt = Path(args.ckpt)
    out = ckpt.parent / "eval" / f"{ckpt.stem}_{args.key}"  # 比如 .../shellgame/eval/last_model/
    model, vae = load_model(ckpt, args.key), VAE()
    acc = {}
    for split in args.splits:
        name = split.split("/")[-1]  # test/n05 -> n05
        acc[name] = evaluate(model, vae, ShellGameDataset(args.root, split), args.num, args.batch,
                             args.steps, args.max_ctx, out, args.save, name)
        print(f"{name}: acc {acc[name]:.3f}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"args": vars(args), "acc": acc}, indent=2))


if __name__ == "__main__":
    main()