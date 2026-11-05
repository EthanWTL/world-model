# 造旋转方块数据：抽 episode -> GPU 渲染 -> Wan2.1 VAE 编码成 latent（格式和 Shell Game 一样：float16 [16, 5K, 24, 32]，
# 归一化 (mu - latents_mean) / latents_std）。不存视频：评测时按 meta.jsonl 里的参数精确重渲染 GT。
#   <root>/train        gap ~ U{0, 2, 4, 6}（统一的训练 gap 混合分布），--train-num 个
#   <root>/test/gXX     每个 gap --test-num 个：0–6 是训练范围内，8–32 更长；seed 和 train 不同（所有配置共用）
#   python build.py --meta                    只写 meta.jsonl（所有 split；已经有了就不重写）
#   python build.py --encode --claim          编码：每个 worker 抢一个没人领的 shard（本地卡和 Slurm 数组可以一起跑）
#   python build.py --check 8                 VAE 往返：渲染 -> 编码 -> 解码，看每个 chunk 末帧（正对）的 cell_acc / psnr
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cube import Scene, episode, render_episode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATA_DIR  # noqa: E402
from vae_io import claim, encode, latent_path, load_vae  # noqa: E402,F401

ROOT = f"{DATA_DIR}/CubeMemory"
TRAIN_GAPS = (0, 2, 4, 6)
TEST_GAPS = (0, 2, 4, 6, 8, 12, 16, 24, 32)


def build_splits(args):
    rng = np.random.default_rng([args.seed, 0])
    train = [episode(rng, int(rng.choice(TRAIN_GAPS)), args.grid) for _ in range(args.train_num)]
    splits = {"train": train}
    for g in TEST_GAPS:
        rng = np.random.default_rng([args.seed, 1, g])
        splits[f"test/g{g:02d}"] = [episode(rng, g, args.grid) for _ in range(args.test_num)]
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--train-num", type=int, default=60000)
    ap.add_argument("--test-num", type=int, default=64)
    ap.add_argument("--meta", action="store_true")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--claim", action="store_true")
    ap.add_argument("--num-shards", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--check", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root)

    if args.meta:
        if (root / "train" / "meta.jsonl").exists():
            sys.exit(f"{root} already has meta.jsonl")
        for name, eps in build_splits(args).items():
            (root / name).mkdir(parents=True, exist_ok=True)
            with open(root / name / "meta.jsonl", "w") as f:
                for i, ep in enumerate(eps):
                    f.write(json.dumps({"id": i, **ep}) + "\n")
            print(name, len(eps), "episodes, chunks", sorted({ep["num_chunks"] for ep in eps}), flush=True)
        (root / "info.json").write_text(json.dumps({
            "source": "auto_research/visual/datagen/cube.py + build.py (torch ray-cast renderer, 4x4 SSAA)",
            "size": [256, 192], "fps": 16, "grid": args.grid, "colors": 8, "seed": args.seed,
            "train_gaps": TRAIN_GAPS, "test_gaps": TEST_GAPS, "test_num": args.test_num,
            "actions": "0 hold, 1 frontal face +1, 2 frontal face -1 (90 deg per chunk)",
            "episode": "chunk 0 hold, chunks 1-3 observe (ends with target frontal), chunk 4 leave, gap chunks walk in "
                       "the back range, last chunk returns; evaluate the last frame",
            "latents": "float16 [16, 5K, 24, 32], Wan2.1 VAE mode, (mu - latents_mean) / latents_std"}, indent=2))

    if args.check:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from metrics import FaceMetrics
        vae, mean, std = load_vae()
        scene, fm = Scene(1, device="cuda"), FaceMetrics(device="cuda")
        rng = np.random.default_rng(123)
        rows = {}
        torch.set_grad_enabled(False)
        for i in range(args.check):
            ep = episode(rng, 2 * (i % 4), args.grid)
            fr = render_episode(scene, ep)
            z = encode(vae, mean, std, [fr])
            full = ((vae.decode(z.float() * std + mean).sample.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 3, 0)  # [T, H, W, 3]
            last = ((vae.decode(z[:, :, -5:].float() * std + mean).sample.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 3, 0)
            ends = [16 + 20 * c for c in range(ep["num_chunks"])]
            gt = torch.from_numpy(fr[ends]).float().cuda() / 255
            faces = torch.tensor(ep["textures"][0]).expand(len(ends), -1, -1, -1)
            front = torch.tensor([p[0] for p in ep["pose"]])
            for name, dec in (("vae_full", full[ends]), ("vae_last_chunk_only", last[-1:])):
                g, f, t = (gt, faces, front) if name == "vae_full" else (gt[-1:], faces[-1:], front[-1:])
                for k, v in fm(dec, g, f, t).items():
                    rows.setdefault((name, k), []).append(v.cpu())
            for k, v in fm(gt, gt, faces, front).items():
                rows.setdefault(("gt_self", k), []).append(v.cpu())
        for (name, k), v in sorted(rows.items()):
            print(f"{name:22s} {k:12s} {torch.cat(v).float().mean():.4f}")

    if args.encode:
        vae, mean, std = load_vae()
        scene = Scene(1, device="cuda")
        splits = ["train"] + [f"test/g{g:02d}" for g in TEST_GAPS]
        metas = {s: [json.loads(line) for line in open(root / s / "meta.jsonl")] for s in splits}
        tag = f"{socket.gethostname()}.{os.getpid()}"
        for shard in range(args.num_shards):
            if args.claim and not claim(root, shard, args.num_shards):
                continue
            t0, done = time.time(), 0
            for s in splits:
                todo = [ep for ep in metas[s] if ep["id"] % args.num_shards == shard
                        and not latent_path(root, s, ep["id"]).exists()]
                groups = {}
                for ep in todo:  # 一个 batch 里帧数一样
                    groups.setdefault(ep["num_frames"], []).append(ep)
                for T, eps in groups.items():
                    bs = max(1, args.batch * 240 // T)
                    for i in range(0, len(eps), bs):
                        b = eps[i:i + bs]
                        z = encode(vae, mean, std, [render_episode(scene, ep) for ep in b]).cpu().numpy()
                        for ep, zi in zip(b, z):
                            out = latent_path(root, s, ep["id"])
                            out.parent.mkdir(parents=True, exist_ok=True)
                            tmp = out.with_name(out.name + f".tmp.{tag}")
                            with open(tmp, "wb") as f:
                                np.save(f, zi)
                            os.replace(tmp, out)
                        done += len(b)
            dt = time.time() - t0
            print(f"[shard {shard}/{args.num_shards}] {done} episodes in {dt / 60:.1f} min ({done / max(dt, 1e-6):.2f}/s)",
                  flush=True)


if __name__ == "__main__":
    main()
