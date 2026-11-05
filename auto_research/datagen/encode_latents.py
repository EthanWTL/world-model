# Encode the rendered Shell Game videos with the frozen Wan2.1 VAE (16 ch, 4x temporal, 8x spatial).
# One file per episode next to its video: <split>/latents/XX/NNNNNN.npy (same numbering as videos/),
# float16 [16, 5(N+3), 24, 32], normalized as (mu - latents_mean) / latents_std, i.e. exactly what the DiT sees.
# Finished files are skipped, so re-running resumes.
#   one GPU   : python shellgame/encode_latents.py
#   many GPUs : the same with --claim on every GPU, all at once (CUDA_VISIBLE_DEVICES=i ... &, torchrun, or
#               sbatch shellgame/encode.slurm): each worker takes the next shard (episodes with id % NUM_SHARDS == s)
#               nobody has claimed yet. Claims live in <root>/logs/encode_claims; once no worker is left running,
#               rm -r that folder to re-open every shard (e.g. after a crash).
import argparse
import json
import os
import socket
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKLWan


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[..., ::-1])
    return np.stack(frames)  # [T, H, W, 3] RGB uint8


def latent_path(root, split, ep):
    return root / split / ep["video"].replace("videos/", "latents/").replace(".mp4", ".npy")


def claim(root, shard, num_shards):
    # O_EXCL: exactly one worker, local or on any Slurm node, manages to create the file, and it gets the shard
    d = root / "logs" / "encode_claims"
    d.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(d / f"{shard:03d}of{num_shards}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{socket.gethostname()} job {os.environ.get('SLURM_JOB_ID', '-')} pid {os.getpid()}\n".encode())
    os.close(fd)
    return True


def listing(d, kind, suffix):
    # d/kind/XX/ 下面已经写好的文件（列目录比一个个 stat 快得多；.tmp 半成品不算）
    out, base = set(), d / kind
    if base.exists():
        for sub in os.listdir(base):
            if (base / sub).is_dir():
                out.update(f"{kind}/{sub}/{f}" for f in os.listdir(base / sub) if f.endswith(suffix))
    return out


def lock(d, ep, steal_after):
    # 一个 episode 一个锁文件（O_EXCL），本地和 Slurm 的 worker 可以一起编；锁太旧还没出 latent = 那个 worker 死了，抢过来
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{ep['id']:06d}"
    for _ in range(2):
        try:
            fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{socket.gethostname()} pid {os.getpid()}\n".encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                if time.time() - f.stat().st_mtime < steal_after:
                    return False
                f.unlink()
            except FileNotFoundError:
                pass
    return False


def encode_items(items, vae, mean, std, pool, root, tag):
    # items: [(split, [ep, ...])]，一个 batch 里帧数一样；CPU 线程提前解码几个 batch
    loads = deque()

    def prefetch(n):
        if n < len(items):
            split, b = items[n]
            loads.append(pool.submit(lambda: np.stack([read_video(root / split / ep["video"]) for ep in b])))

    for n in range(4):
        prefetch(n)
    for n, (split, batch) in enumerate(items):
        video = loads.popleft().result()
        prefetch(n + 4)
        x = torch.from_numpy(video).cuda().permute(0, 4, 1, 2, 3).float() / 127.5 - 1  # [B, 3, T, H, W]
        with torch.no_grad():
            z = ((vae.encode(x).latent_dist.mode() - mean) / std).half().cpu().numpy()
        for ep, zi in zip(batch, z):
            out = latent_path(root, split, ep)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + f".tmp.{tag}")
            with open(tmp, "wb") as f:
                np.save(f, zi)
            os.replace(tmp, out)


def follow(args, root, metas, vae, mean, std, pool, tag):
    # 边渲染边编码：反复扫一遍，哪些视频已经渲染好、还没有 latent、没人锁着，就锁上编掉；全部都有 latent 了才退出
    t0, done = time.time(), 0
    while True:
        ready, left = [], 0
        for split in args.splits:
            vids, lats = listing(root / split, "videos", ".mp4"), listing(root / split, "latents", ".npy")
            for ep in metas[split]:
                if ep["video"].replace("videos/", "latents/").replace(".mp4", ".npy") not in lats:
                    left += 1
                    if ep["video"] in vids:
                        ready.append((split, ep))
        if left == 0:
            break
        groups = {}
        for split, ep in ready:
            groups.setdefault((split, ep["num_frames"]), []).append(ep)
        items = []
        for (split, _), eps in groups.items():
            got = []
            for ep in eps:
                if lock(root / "logs" / "encode_locks" / split, ep, args.steal_after):
                    got.append(ep)
                    if len(got) == args.batch:
                        items.append((split, got))
                        got = []
                        if len(items) >= 32:
                            break
            if got:
                items.append((split, got))
            if len(items) >= 32:
                break
        if not items:
            print(f"[follow] {left} left, nothing ready yet", flush=True)
            time.sleep(60)
            continue
        encode_items(items, vae, mean, std, pool, root, tag)
        done += sum(len(b) for _, b in items)
        print(f"[follow] {done} encoded by me, {done / (time.time() - t0):.2f}/s, {left} left before this round", flush=True)
    print(f"[follow] all done: {done} episodes by me in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/project/sds-rise/ethan/Dataset/ShellGame")
    ap.add_argument("--splits", nargs="+", default=["train"] + [f"test/n{n:02d}" for n in range(5, 31)])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="encode only the first LIMIT episodes of each split")
    ap.add_argument("--num-shards", type=int, default=64, help="shard s = episodes with id %% NUM_SHARDS == s, in every split")
    ap.add_argument("--claim", action="store_true", help="encode only shards no other worker has claimed")
    ap.add_argument("--follow", action="store_true", help="auto_research: encode while rendering, per-episode locks")
    ap.add_argument("--steal-after", type=float, default=1800, help="--follow: a lock this old without a latent is stale")
    args = ap.parse_args()
    root = Path(args.root)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    tag = f"{socket.gethostname()}.{os.getpid()}"

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae",
                                           torch_dtype=torch.float32).cuda().eval()
    mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda").view(1, 16, 1, 1, 1)
    metas = {split: [json.loads(line) for line in open(root / split / "meta.jsonl")][:args.limit] for split in args.splits}

    pool = ThreadPoolExecutor(4)
    if args.follow:
        return follow(args, root, metas, vae, mean, std, pool, tag)
    for shard in range(args.num_shards):
        if args.claim and not claim(root, shard, args.num_shards):
            continue
        t0, done = time.time(), 0
        for split in args.splits:
            todo = [ep for ep in metas[split] if ep["id"] % args.num_shards == shard
                    and not latent_path(root, split, ep).exists() and (root / split / ep["video"]).exists()]
            # 课程数据里 N 不一样长：只把帧数相同的 episode 放进一个 batch（auto_research 的改动）
            groups = {}
            for ep in todo:
                groups.setdefault(ep["num_frames"], []).append(ep)
            batches = [g[i:i + args.batch] for g in groups.values() for i in range(0, len(g), args.batch)]
            loads = deque()  # decode a few batches ahead on CPU threads while the GPU encodes

            def prefetch(n):
                if n < len(batches):
                    loads.append(pool.submit(lambda b: np.stack([read_video(root / split / ep["video"]) for ep in b]), batches[n]))

            for n in range(4):
                prefetch(n)
            for n, batch in enumerate(batches):
                video = loads.popleft().result()
                prefetch(n + 4)
                x = torch.from_numpy(video).cuda().permute(0, 4, 1, 2, 3).float() / 127.5 - 1  # [B, 3, T, H, W]
                with torch.no_grad():
                    z = ((vae.encode(x).latent_dist.mode() - mean) / std).half().cpu().numpy()
                for ep, zi in zip(batch, z):
                    out = latent_path(root, split, ep)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_name(out.name + f".tmp.{tag}")
                    with open(tmp, "wb") as f:
                        np.save(f, zi)
                    os.replace(tmp, out)
                done += len(batch)
                if n % 50 == 49:
                    print(f"[shard {shard}/{args.num_shards}] {split}: {n + 1}/{len(batches)} batches", flush=True)
        dt = time.time() - t0
        print(f"[shard {shard}/{args.num_shards}] done: {done} episodes in {dt / 60:.1f} min, {done / dt:.2f}/s", flush=True)


if __name__ == "__main__":
    main()
