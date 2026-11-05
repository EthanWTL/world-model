#!/usr/bin/env python3
"""Checks the setup before a run (CPU, about a minute): Python and package versions against env/requirements.lock.txt,
the GPUs, ffmpeg, the VAE files (and that they load offline), the two locations (set, writable, free space), the
dataset's state, and the wandb key. With --data-only: only whether every episode of the dataset has its latent.

    ./run.sh check                  # the same environment variables as the runs (recommended)
    ./run.sh check --data-only      # after the data stage: READY when the dataset is complete

Exit code 0 = everything needed is in place (warnings may still be printed); 1 = something must be fixed first.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
SPLITS = ["train"] + [f"test/n{n:02d}" for n in (0, 2, 4, 8, 12, 16, 24)]
COUNTS = {"train": 60000, **{s: 64 for s in SPLITS[1:]}}
VAE_FILES = {"vae/config.json": 724, "vae/diffusion_pytorch_model.safetensors": 507591892}
NEED_GB = {"VM_DATA_DIR": 80, "VM_OUT_DIR": 120}
bad, warn = [], []


def ok(msg):
    print(f"  ok    {msg}")


def fail(msg):
    bad.append(msg)
    print(f"  FAIL  {msg}")


def note(msg):
    warn.append(msg)
    print(f"  warn  {msg}")


def dataset(strict):
    # strict: an incomplete dataset is a failure (--data-only, after the data stage); otherwise only a warning
    root = Path(os.environ["VM_DATA_DIR"]) / "RubikMemory"
    if not (root / "train" / "meta.jsonl").exists():
        (fail if strict else note)(f"dataset: no meta.jsonl under {root} yet (the data stage writes it)")
        return
    for s in SPLITS:
        meta = root / s / "meta.jsonl"
        n_meta = sum(1 for _ in open(meta)) if meta.exists() else 0
        lat = root / s / "latents"
        n_lat = sum(len([f for f in os.listdir(lat / d) if f.endswith(".npy")]) for d in os.listdir(lat)) if lat.exists() else 0
        msg = f"dataset {s}: {n_lat} / {n_meta} latents"
        if n_meta and n_lat == n_meta:
            ok(msg)
        else:
            (fail if strict else note)(msg + " (the data stage is not finished)")
        if n_meta != COUNTS[s]:
            note(f"{s}: meta has {n_meta} episodes, the experiment uses {COUNTS[s]} (fine for the smoke test only)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-only", action="store_true")
    args = ap.parse_args()
    for v in ("VM_DATA_DIR", "VM_OUT_DIR"):
        if not os.environ.get(v) or os.environ[v].startswith("/path/to"):
            fail(f"{v} is not set (config.env)")
    if bad:
        sys.exit(1)
    if args.data_only:
        print("dataset")
        dataset(strict=True)
        print("READY: the dataset is complete" if not bad else "NOT READY: the dataset is incomplete")
        sys.exit(1 if bad else 0)

    print("python and packages")
    if sys.version_info[:2] == (3, 11):
        ok(f"python {sys.version.split()[0]}")
    else:
        fail(f"python {sys.version.split()[0]} (3.11 expected: env/setup_env.sh)")
    want = {}
    for line in open(HERE / "env" / "requirements.lock.txt"):
        if "==" in line and not line.startswith("#"):
            name, ver = line.strip().split("==")
            want[name] = ver
    diff = {}
    for name, ver in want.items():
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            have = None
        if have != ver:
            diff[name] = (have, ver)
    ok(f"{len(want)} pinned packages match") if not diff else fail(f"versions differ (installed, wanted): {diff}")
    try:
        import torch
        tv = torch.__version__
        ok(f"torch {tv}") if tv.startswith("2.8.0") else fail(f"torch {tv} (2.8.0 expected)")
    except Exception as e:  # noqa: BLE001
        fail(f"torch does not import: {e}")
        torch = None

    print("GPUs")
    if torch is not None:
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n == 0:
            note("no GPU visible here (fine on a login / CPU node; the runs need GPUs)")
        else:
            names = sorted({torch.cuda.get_device_name(i) for i in range(n)})
            mem = min(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 2 ** 30
            ok(f"{n} x {', '.join(names)}, {mem:.0f} GB each")
            if mem < 40:
                fail("each GPU needs >= 40 GB (a fine-tune and a base evaluation share the first GPU)")
            cap = torch.cuda.get_device_capability(0)
            if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
                fail(f"torch has no kernels for sm_{cap[0]}{cap[1]}: rebuild the environment (env/setup_env.sh, TORCH_CUDA=cu128)")
            if n not in (1, 2, 4, 8):
                note(f"{n} GPUs visible: the base needs 1, 2, 4 or 8 of them (global batch 16)")

    print("ffmpeg (videos)")
    try:
        import imageio_ffmpeg
        ok(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as e:  # noqa: BLE001
        fail(f"imageio-ffmpeg: {e}")

    print("VAE")
    vae = Path(os.environ.get("VM_VAE", HERE / "models" / "Wan2.1-T2V-1.3B-Diffusers"))
    sizes = {f: (vae / f).stat().st_size if (vae / f).exists() else None for f in VAE_FILES}
    if all(sizes[f] == s for f, s in VAE_FILES.items()):
        ok(f"{vae}: both files, sizes match")
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            from diffusers import AutoencoderKLWan
            m = AutoencoderKLWan.from_pretrained(str(vae), subfolder="vae")
            ok(f"loads offline ({sum(p.numel() for p in m.parameters()) / 1e6:.1f}M parameters)")
        except Exception as e:  # noqa: BLE001
            fail(f"VAE does not load: {e}")
    else:
        fail(f"{vae}: files missing or wrong size {sizes} (./download.sh)")

    print("locations")
    for v in ("VM_DATA_DIR", "VM_OUT_DIR"):
        p = Path(os.environ[v])
        try:
            p.mkdir(parents=True, exist_ok=True)
            t = p / ".write_test"
            t.write_text("x")
            t.unlink()
            free = shutil.disk_usage(p).free / 2 ** 30
            (ok if free >= NEED_GB[v] else note)(f"{v}={p}: writable, {free:.0f} GB free (needs ~{NEED_GB[v]} GB)")
        except OSError as e:
            fail(f"{v}={p}: {e}")

    print("dataset")
    dataset(strict=False)

    print("wandb")
    if os.environ.get("WANDB_API_KEY"):
        ok("WANDB_API_KEY set: runs log to wandb (project world-auto-visual)")
    else:
        note("no WANDB_API_KEY: wandb off; every logged value is in <run>/log.jsonl anyway")

    print()
    print("READY" if not bad else f"NOT READY: {len(bad)} item(s) to fix")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
