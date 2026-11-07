# 造魔方数据（rubik.py 的 episode + 渲染，用户 2026-11-03 看过小样："都对 直接开始做"）：抽 episode -> GPU 渲染 -> Wan2.1 VAE 编码
#   latent 格式和旋转方块一样：float16 [16, 5K, 24, 32]，(mu - latents_mean) / latents_std；不存视频，评测时按 meta.jsonl 精确重渲染
#   <root>/train        打乱步数 N ~ U{0..8}（K = 7 + N 个 chunk），--train-num 个
#   <root>/test/nXX     每个 N --test-num 个：0–8 在训练范围内，12–24 更长；seed 和 train 不同
#   python build_rubik.py --meta                 只写 meta.jsonl（已经有了就不重写）
#   python build_rubik.py --encode --claim       编码：每个 worker 抢没人领的 shard（本地卡和 Slurm 数组可以一起跑）
#   python build_rubik.py --encode               不抢 shard，把所有还缺的 latent 补齐（收尾：被 Slurm 超时打断的 shard）
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vae_io import claim, encode, latent_path, load_vae  # noqa: E402
from rubik import ACTIONS, DISPLAY, FPS, H, W, RubikScene, episode, render_episode  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATA_DIR  # noqa: E402

ROOT = f"{DATA_DIR}/RubikMemory"
TRAIN_MOVES = tuple(range(9))                # 0..8
TEST_MOVES = (0, 2, 4, 8, 12, 16, 24)


def splits_of():
    return ["train"] + [f"test/n{n:02d}" for n in TEST_MOVES]


def build_splits(args):
    rng = np.random.default_rng([args.seed, 0])
    splits = {"train": [episode(rng, int(rng.choice(TRAIN_MOVES)), args.n, args.grid) for _ in range(args.train_num)]}
    for n in TEST_MOVES:
        rng = np.random.default_rng([args.seed, 1, n])
        splits[f"test/n{n:02d}"] = [episode(rng, n, args.n, args.grid) for _ in range(args.test_num)]
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--train-num", type=int, default=60000)
    ap.add_argument("--test-num", type=int, default=64)
    ap.add_argument("--meta", action="store_true")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--claim", action="store_true")
    ap.add_argument("--num-shards", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--splits", nargs="+", default=None)  # 只编这些 split（例：test/n00 test/n24 或 test = 所有测试 split）
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
            "source": "auto_research/visual/datagen/rubik.py + build_rubik.py (torch ray-cast renderer, 4x4 SSAA)",
            "size": [W, H], "fps": FPS, "n": args.n, "grid": args.grid, "colors": 8, "seed": args.seed,
            "train_moves": TRAIN_MOVES, "test_moves": TEST_MOVES, "train_num": args.train_num, "test_num": args.test_num,
            "actions": ACTIONS, "display": [ACTIONS[a] for a in DISPLAY],
            "episode": "chunk 0 hold, chunks 1-3 display (y y x), N random quarter face turns (never the inverse of the "
                       "previous one), final display (y y x); K = 7 + N chunks; evaluate the end frames of the last "
                       "scramble chunk and the 3 final display chunks",
            "latents": "float16 [16, 5K, 24, 32], Wan2.1 VAE mode, (mu - latents_mean) / latents_std"}, indent=2))

    if args.encode:
        torch.set_grad_enabled(False)
        vae, mean, std = load_vae()
        scene = RubikScene(args.n, device="cuda")
        todo_splits = [s for s in splits_of() if not args.splits or any(s == x or s.startswith(x + "/") for x in args.splits)]
        metas = {s: [json.loads(line) for line in open(root / s / "meta.jsonl")] for s in todo_splits}
        tag = f"{socket.gethostname()}.{os.getpid()}"
        for shard in range(args.num_shards):
            if args.claim and not claim(root, shard, args.num_shards):
                continue
            t0, done = time.time(), 0
            for s in todo_splits:
                todo = [ep for ep in metas[s] if ep["id"] % args.num_shards == shard
                        and not latent_path(root, s, ep["id"]).exists()]
                groups = {}
                for ep in todo:  # 一个 batch 里帧数一样
                    groups.setdefault(ep["num_frames"], []).append(ep)
                for T, eps in sorted(groups.items()):
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
            print(f"{time.strftime('%F %T')} [shard {shard}/{args.num_shards}] {done} episodes in {dt / 60:.1f} min "
                  f"({done / max(dt, 1e-6):.2f}/s)", flush=True)


if __name__ == "__main__":
    main()
