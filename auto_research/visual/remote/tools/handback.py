#!/usr/bin/env python3
"""The project side of a hand-back of rubik_memory_remote (local only, never shipped):

    python tools/handback.py fetch --folder rubik_r1_XXXXXXXXXXXXXX [--repo riselab/rubik_memory]
    python tools/handback.py eval  --folder rubik_r1_XXXXXXXXXXXXXX [--gpu 0] [--videos 2]

fetch: downloads <folder>/ of the (public) Hugging Face repo into DEST/<folder>/ (DEST = the local runs' output root
       /remote, i.e. Finetune/world/auto_research/visual/remote/), then checks every file of MANIFEST.json (size, sha256).
eval:  the evaluation the remote side does not run: for every inference/<run>/<ckpt>/ with a done.json,
       rubik_eval.py --saved -> DEST/<folder>/<run>/eval/<ckpt>_ema.json (+ videos next to it), then the summary table
       (summary.py --task rubik with VM_OUT_DIR=DEST/<folder>) -> DEST/<folder>/summary_rubik.txt. Needs the local
       dataset's test splits (the same episodes: seed 0) and a GPU of ours (nvidia-smi first).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VISUAL = HERE.parent.parent
sys.path.insert(0, str(VISUAL))
from paths import OUT_DIR  # noqa: E402

DEST = Path(OUT_DIR) / "remote"
PY = sys.executable


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fetch(args):
    from huggingface_hub import snapshot_download
    os.environ.pop("HF_HUB_OFFLINE", None)
    snapshot_download(args.repo, repo_type="model", allow_patterns=[f"{args.folder}/*"], local_dir=DEST, max_workers=4)
    root = DEST / args.folder
    manifest = json.loads((root / "MANIFEST.json").read_text())
    bad = []
    for rel, want in manifest.items():
        p = root / rel
        if not p.exists() or p.stat().st_size != want["size"] or sha256(p) != want["sha256"]:
            bad.append(rel)
    print(f"{root}: {len(manifest) - len(bad)} / {len(manifest)} files match MANIFEST.json")
    if bad:
        raise SystemExit(f"mismatch: {bad[:10]}")


def evaluate(args):
    root = DEST / args.folder
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1"}
    for done in sorted((root / "inference").glob("*/*/done.json")):
        job = done.parent
        run, ckpt = job.parent.name, job.name
        stem = ckpt.split("_ctx")[0]
        ctx = f"_ctx{ckpt.split('_ctx')[1]}" if "_ctx" in ckpt else ""
        out = root / run / "eval" / f"{stem}_ema{ctx}.json"
        if out.exists():
            print(f"{run}/{ckpt}: evaluated already")
            continue
        # VAE 上限只算一次（base 40K、KV 全留那个 job）；视频只给最终的 checkpoint
        vae = run == "rubik_base" and ckpt == "ema_040000_ctx-1"
        videos = args.videos if (ckpt.startswith("ckpt_012000") or run == "rubik_base") else 0
        print(f"{run}/{ckpt} -> {out} (vae {vae}, videos {videos})", flush=True)
        subprocess.run([PY, "-u", "rubik_eval.py", "--saved", str(job), "--out", str(out), "--videos", str(videos),
                        "--gpu", "0"] + ([] if vae else ["--no-vae"]), cwd=VISUAL,
                       env={**env, "CUDA_VISIBLE_DEVICES": str(args.gpu)}, check=True)
    subprocess.run([PY, "summary.py", "--task", "rubik", "--out", str(root / "summary_rubik.txt")], cwd=VISUAL,
                   env={**env, "VM_OUT_DIR": str(root)}, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "eval"])
    ap.add_argument("--folder", required=True)
    ap.add_argument("--repo", default="riselab/rubik_memory")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--videos", type=int, default=2)
    args = ap.parse_args()
    (fetch if args.cmd == "fetch" else evaluate)(args)


if __name__ == "__main__":
    main()
