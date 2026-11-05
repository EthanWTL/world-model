#!/usr/bin/env python3
"""Hand-back: gathers the results of the pipeline into one folder and uploads it to a public Hugging Face model repo
(resumable). With UPLOAD_DATA=1 (or --data) also the dataset, as tar shards, to a public dataset repo.

    HF_TOKEN=hf_... ./run.sh handback                     # results, ~45 GB
    HF_TOKEN=hf_... UPLOAD_DATA=1 ./run.sh handback       # results + the dataset (~75 GB more; needs as much free space
                                                          #   under $VM_OUT_DIR/handback for the tar shards)
    ./run.sh handback --dry-run                           # gather and list only, no upload

Repos (public; created when missing): HANDBACK_REPO (default riselab/rubik_memory, model repo) and DATA_REPO (default
riselab/rubik_memory_data, dataset repo). The token needs write access to the riselab organization.
The folder inside the repo: HANDBACK_FOLDER (default rubik_r1_<YYYYmmddHHMMSS of the first handback>; kept in
$VM_OUT_DIR/handback/FOLDER so that a rerun continues the same folder).

<folder>/ on the Hub:
  README.md, MANIFEST.json      what is inside; every file with size and sha256
  logs/                         every stage log
  rubik_base/                   log.jsonl (training losses), ema_020000.pt, ema_040000.pt
  rk_<config>/                  log.jsonl, info.json, ckpt_012000.pt
  inference/<run>/<checkpoint>/ done.json, nXX/{free.npy, tf.npy, ids.json}: the generated latents of every test split
  data/                         info.json of the dataset, sha256 of every meta.jsonl, latent counts
No evaluation runs on the remote side: the project scores the generated latents itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import tarfile
import time
from pathlib import Path

OUT = Path(os.environ["VM_OUT_DIR"])
DATA = Path(os.environ["VM_DATA_DIR"]) / "RubikMemory"
STAGE = OUT / "handback" / "stage"
def keep(run_dir: Path) -> list[str]:
    # rubik_base: log.jsonl + the final EMA and the one at half its step (40000 and 20000; the smoke test: 20 and 10);
    # rk_*: log.jsonl, info.json + the final checkpoint (12000)
    if run_dir.name == "rubik_base":
        steps = sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("ema_*.pt"))
        picks = [s for s in steps if steps and s in (steps[-1] // 2, steps[-1])]
        return ["log.jsonl"] + [f"ema_{s:06d}.pt" for s in picks]
    steps = sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("ckpt_*.pt"))
    return ["log.jsonl", "info.json"] + ([f"ckpt_{steps[-1]:06d}.pt"] if steps else [])


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def place(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)  # same filesystem: no copy
    except OSError:
        shutil.copy2(src, dst)


def tar_dir(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tar.tmp")
    with tarfile.open(tmp, "w") as t:
        t.add(src, arcname=src.name)
    tmp.rename(dst)


def gather(folder: str) -> Path:
    root = STAGE / folder
    for run_dir in sorted(p for p in OUT.iterdir() if p.is_dir() and (p.name == "rubik_base" or p.name.startswith("rk_"))):
        for name in keep(run_dir):
            if (run_dir / name).exists():
                place(run_dir / name, root / run_dir.name / name)
    inf = OUT / "inference"
    if inf.exists():
        for f in sorted(inf.rglob("*")):
            if f.is_file() and not f.name.endswith(".tmp"):
                place(f, root / "inference" / f.relative_to(inf))
    if (OUT / "logs").exists():
        for f in (OUT / "logs").iterdir():
            if f.is_file():
                place(f, root / "logs" / f.name)
    d = {"counts": {}, "meta_sha256": {}}
    for meta in sorted(DATA.glob("**/meta.jsonl")):
        split = str(meta.parent.relative_to(DATA))
        lat = meta.parent / "latents"
        d["counts"][split] = {"meta": sum(1 for _ in open(meta)),
                              "latents": sum(len(os.listdir(lat / s)) for s in os.listdir(lat)) if lat.exists() else 0}
        d["meta_sha256"][split] = sha256(meta)
    (root / "data").mkdir(parents=True, exist_ok=True)
    if (DATA / "info.json").exists():
        place(DATA / "info.json", root / "data" / "info.json")
    (root / "data" / "dataset_state.json").write_text(json.dumps(d, indent=1))
    try:
        import torch
        gpu = f"{torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "-"
    except Exception:  # noqa: BLE001
        gpu = "-"
    runs = sorted(p.name for p in root.iterdir() if p.is_dir() and (p.name.startswith("rk_") or p.name == "rubik_base"))
    jobs = sorted(str(p.parent.relative_to(root / "inference")) for p in (root / "inference").rglob("done.json")) \
        if (root / "inference").exists() else []
    (root / "README.md").write_text(
        f"# {folder}\n\nHand-back of the rubik_memory_remote package (2x2 Rubik's cube, visual memory in native fast "
        f"weights): checkpoints, training logs and the generated latents of the test splits (no evaluation).\n"
        f"Host {socket.gethostname()}, GPUs {gpu}, gathered {time.strftime('%F %T')}.\n\nRuns: {', '.join(runs)}\n\n"
        f"Inference ({len(jobs)}): {', '.join(jobs)}\n\nEvery file with size and sha256: MANIFEST.json.\n")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST.json" and ".cache" not in p.parts)
    manifest = {str(p.relative_to(root)): {"size": p.stat().st_size, "sha256": sha256(p)} for p in files}
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    total = sum(v["size"] for v in manifest.values()) / 2 ** 30
    print(f"gathered {len(manifest)} files, {total:.1f} GB -> {root}")
    for k, v in manifest.items():
        if v["size"] > 2 ** 27:
            print(f"  {k}: {v['size'] / 2 ** 30:.2f} GB")
    return root


def pack_data(folder: str) -> Path:
    root = OUT / "handback" / "data_stage" / folder / "RubikMemory"
    for meta in sorted(DATA.glob("**/meta.jsonl")):
        split = meta.parent.relative_to(DATA)
        place(meta, root / split / "meta.jsonl")
        lat = meta.parent / "latents"
        for sub in sorted(os.listdir(lat)) if lat.exists() else []:
            tar_dir(lat / sub, root / split / f"latents_{sub}.tar")  # tar xf -> <sub>/<id>.npy; put it under latents/
    if (DATA / "info.json").exists():
        place(DATA / "info.json", root / "info.json")
    print(f"packed the dataset -> {root}")
    return root.parent.parent


def upload(api, repo: str, repo_type: str, folder_path: Path):
    api.create_repo(repo, repo_type=repo_type, private=False, exist_ok=True)
    info = api.repo_info(repo, repo_type=repo_type)
    if getattr(info, "private", False):
        print(f"warning: {repo} exists and is private; the project expects a public repo")
    print(f"uploading {folder_path} -> {repo_type} repo {repo} (resumable: rerun the same command after an interruption)")
    api.upload_large_folder(repo_id=repo, folder_path=str(folder_path), repo_type=repo_type)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="store_true", default=os.environ.get("UPLOAD_DATA") == "1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    (OUT / "handback").mkdir(parents=True, exist_ok=True)
    ff = OUT / "handback" / "FOLDER"
    folder = os.environ.get("HANDBACK_FOLDER") or (ff.read_text().strip() if ff.exists() else f"rubik_r1_{time.strftime('%Y%m%d%H%M%S')}")
    ff.write_text(folder + "\n")
    root = gather(folder)
    data_root = pack_data(folder) if args.data else None
    if args.dry_run:
        print("dry run: nothing uploaded")
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (a token with write access to the riselab organization)")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    repo = os.environ.get("HANDBACK_REPO", "riselab/rubik_memory")
    upload(api, repo, "model", root.parent)
    print(f"results: https://huggingface.co/{repo}/tree/main/{folder}")
    if data_root is not None:
        drepo = os.environ.get("DATA_REPO", "riselab/rubik_memory_data")
        upload(api, drepo, "dataset", data_root)
        print(f"dataset: https://huggingface.co/datasets/{drepo}/tree/main/{folder}")
    print(f"hand-back done: folder {folder}")


if __name__ == "__main__":
    main()
