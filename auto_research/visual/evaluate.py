# 回访评测：观察阶段给 4 个 GT chunk（四个面都看过，最后目标面正对），之后只按动作生成；
# 目标面转回正对的那一帧（返回 chunk 的最后一帧）和原始图案比（指标见 metrics.py）
#   free ：只给观察的 4 个 GT chunk，后面全部自己生成（主协议；返回面的真实图像从来不输入）
#   tf   ：除了返回 chunk 都给 GT（间隔里目标面看不见，所以也没输入返回面），只生成返回 chunk = 只看记忆、没有生成误差累积
#          tf1_*：σ=1（纯噪声）一步预测的 x0 = 只靠记忆猜的平均画面（不采样，便宜）
#   vae_*：GT latent 直接解码 = VAE 的上限（基础画质）
# 每个 gap 还记：gap 的帧数、chunk 数、间隔里的写入次数（= gap 的 chunk 数）
#   python evaluate.py --ckpt <OUT>/<run>/ckpt_012000.pt --key ema
#   python evaluate.py --ckpt .../visual/base/ema_020000.pt --max-ctx -1        （base：KV cache 全留 = 有历史的上限）
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parent / "datagen"))
from build import TEST_GAPS  # noqa: E402
from cube import H, W, Scene, render_episode  # noqa: E402
from data import TASKS, CubeDataset  # noqa: E402
from metrics import FaceMetrics  # noqa: E402
from sampling import build_from_ckpt, randn, rollout, sample_chunk, sigma_schedule  # noqa: E402,F401

from paths import OUT_DIR as OUT  # noqa: E402
from paths import VAE as VAE_REPO  # noqa: E402


class VAE:
    def __init__(self):
        from diffusers import AutoencoderKLWan
        self.vae = AutoencoderKLWan.from_pretrained(VAE_REPO, subfolder="vae",
                                                    torch_dtype=torch.float32).cuda().eval()
        self.mean = torch.tensor(self.vae.config.latents_mean, device="cuda").view(1, 16, 1, 1, 1)
        self.std = torch.tensor(self.vae.config.latents_std, device="cuda").view(1, 16, 1, 1, 1)

    @torch.no_grad()
    def decode(self, z):
        # z: [B, 16, T, 24, 32] -> [B, 4T-3, 192, 256, 3] float [0, 1]
        x = self.vae.decode(z.float() * self.std + self.mean).sample
        return ((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 4, 1)


class Evaluator:
    def __init__(self, gaps=(0, 2, 6), num=32, batch=16, steps=20, modes=("tf",), seed=0, videos=0, video_dir=None):
        self.vae = VAE()
        self.scene = Scene(1, device="cuda")
        self.fm = FaceMetrics(self.scene, device="cuda")
        self.data = {g: CubeDataset(f"test/g{g:02d}") for g in gaps}
        self.num, self.batch, self.steps, self.modes, self.seed = num, batch, steps, modes, seed
        self.videos, self.video_dir = videos, video_dir

    def last_frames(self, eps):
        # 每个 episode 返回 chunk 的最后一帧（目标面正对），精确渲染
        return torch.cat([self.scene.render(np.array([[-ep["target"] * np.pi / 2]]), ep["textures"]) for ep in eps])

    def save_video(self, ep, z, mode, gap):
        # GT（精确渲染）| 生成；生成那一半顶上一条：绿 = 输入的 GT chunk，红 = 自己生成的，黄 = 返回 chunk
        import imageio.v2 as imageio
        gt = render_episode(self.scene, ep)
        gen = (self.vae.decode(z[None])[0].cpu().numpy() * 255).round().astype(np.uint8)
        n_ctx = ep["num_chunks"] - 1 if mode == "tf" else 4
        for f in range(len(gen)):
            c = 0 if f <= 16 else 1 + (f - 17) // 20
            gen[f, :4] = (40, 200, 60) if c < n_ctx else (240, 200, 30) if c == ep["num_chunks"] - 1 else (220, 40, 40)
        out = Path(self.video_dir) / f"g{gap:02d}_ep{ep['id']:03d}_{mode}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(out, np.concatenate([gt, gen], 2), fps=16, quality=9, macro_block_size=1)

    @torch.no_grad()
    def run(self, model, per_episode=False):
        was = model.training
        model.eval()
        res, per = {}, {}
        for gap, ds in self.data.items():
            n = min(self.num, len(ds))
            if n == 0:  # 这个 gap 还没编码
                continue
            acc = {}
            # KV 全留（base）时长 episode 的 cache 很大：按 chunk 数缩小 batch（噪声按 episode 定，不影响结果）
            bs = self.batch if model.max_ctx is not None else max(1, self.batch * 12 // ds.eps[0]["num_chunks"])
            for i in range(0, n, bs):
                idx = list(range(i, min(i + bs, n)))
                b = default_collate([ds[j] for j in idx])
                eps = [ds.eps[j] for j in idx]
                x0, a = b["latent"].cuda(), b["actions"].cuda()
                gt = self.last_frames(eps)
                faces = torch.tensor([ep["textures"][0] for ep in eps])
                target = torch.tensor([ep["target"] for ep in eps])
                for k, v in self.fm(self.vae.decode(x0[:, :, -5:])[:, -1], gt, faces, target).items():
                    acc.setdefault(f"vae_{k}", []).append(v.cpu())
                for mode in self.modes:
                    n_ctx = a.shape[1] - 1 if mode == "tf" else 4
                    keep = i == 0 and self.videos > 0
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        x, x0_hat, z = rollout(model, x0, a, n_ctx, self.steps,
                                               [self.seed * 100000 + ep["id"] for ep in eps], keep)
                    for k, v in self.fm(self.vae.decode(x)[:, -1], gt, faces, target).items():
                        acc.setdefault(f"{mode}_{k}", []).append(v.cpu())
                    if mode == "tf":
                        for k, v in self.fm(self.vae.decode(x0_hat)[:, -1], gt, faces, target).items():
                            acc.setdefault(f"tf1_{k}", []).append(v.cpu())
                    if keep:
                        for j in range(min(self.videos, len(eps))):
                            self.save_video(eps[j], z[j], mode, gap)
            for k, v in acc.items():
                v = torch.cat(v).float()
                res[f"g{gap:02d}_{k}"] = v.mean().item()
                if per_episode:
                    per[f"g{gap:02d}_{k}"] = v.tolist()
            ep = ds.eps[0]
            res[f"g{gap:02d}_gap_chunks"], res[f"g{gap:02d}_gap_frames"] = ep["gap"], ep["gap_frames"]
            res[f"g{gap:02d}_writes_gap"], res[f"g{gap:02d}_chunks"] = ep["writes_gap"], ep["num_chunks"]
        model.train(was)
        return (res, per) if per_episode else res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--key", default=None)                      # ckpt 里用 "ema" 还是 "model"（默认 ema）
    ap.add_argument("--max-ctx", type=int, default=-2)          # -2 = ckpt 里的设置，-1 = KV cache 全留
    ap.add_argument("--gaps", type=int, nargs="+", default=list(TEST_GAPS))
    ap.add_argument("--modes", nargs="+", default=["tf", "free"])
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--videos", type=int, default=2)            # 每个 gap 存前几个 episode 的对比视频
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    model, cargs, step = build_from_ckpt(args.ckpt, args.key, args.max_ctx)
    ctx = "" if args.max_ctx == -2 else f"_ctx{args.max_ctx}"
    out = Path(args.out or Path(args.ckpt).parent / "eval" / f"{Path(args.ckpt).stem}_{args.key or 'ema'}{ctx}.json")
    if cargs.get("task", "cube") == "rubik":  # 魔方：--gaps 当打乱步数 N 用（默认 = 所有测试 N）
        from rubik_eval import RubikEvaluator
        moves = args.gaps if args.gaps != list(TEST_GAPS) else list(TASKS["rubik"]["test"])
        ev = RubikEvaluator(tuple(moves), args.num, args.batch, args.steps, tuple(args.modes), videos=args.videos,
                            video_dir=out.with_suffix(""))
    else:
        ev = Evaluator(args.gaps, args.num, args.batch, args.steps, tuple(args.modes), videos=args.videos,
                       video_dir=out.with_suffix(""))
    res, per = ev.run(model, per_episode=True)
    for k, v in res.items():
        if any(s in k for s in ("cell_acc", "psnr", "lpips", "wrong_face", "yaw_err", "_acc", "_hid", "_vis")):
            print(f"{k}: {v:.4f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "step": step, "res": res, "per_episode": per}, indent=1))


if __name__ == "__main__":
    main()
