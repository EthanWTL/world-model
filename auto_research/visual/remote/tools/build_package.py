#!/usr/bin/env python3
"""Builds the remote package: package/ (INSTRUCTIONS.md, run.sh, config.env, check_setup.py, download.sh, upload.py,
env/) + the code of auto_research/visual (the same files the local runs use) -> dist/rubik_memory_remote/ and
dist/rubik_memory_remote.zip.

    python tools/build_package.py

Only dataset creation, training and inference are shipped (user, 2026-11-03: "不能有evaluation 只能有dataset creation
training 和inference eval这些我们拿回来跑"): the evaluation modules (evaluate, rubik_eval, metrics, summary, check) stay
here. paths.py is replaced by a version without local defaults (the locations come only from run.sh / config.env), local
paths in comments become $VM_DATA_DIR / $VM_OUT_DIR, and the build refuses to finish when a shipped file still contains
a local path, host or user name, imports an evaluation module at top level, or does not import from the dist folder
alone, or when config.env lacks its /path/to placeholders.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = HERE.parent
VISUAL = REMOTE.parent
NAME = "rubik_memory_remote"
DIST = REMOTE / "dist" / NAME
CODE = ["wm.py", "train_base.py", "train.py", "infer.py", "sampling.py", "data.py", "normuon.py"]
DATAGEN = ["rubik.py", "build_rubik.py", "vae_io.py"]
EVAL_MODULES = ("evaluate", "rubik_eval", "metrics", "summary", "check", "cube", "build", "rubik_sample")
SUBST = [("/project/sds-rise/ethan/Finetune/world/auto_research/visual", "$VM_OUT_DIR"),
         ("/project/sds-rise/ethan/Dataset", "$VM_DATA_DIR"),
         ("/project/sds-rise/ethan/miniforge3/envs/world/bin/python", "python")]
FORBIDDEN = ["/project/", "/sfs/", "sds-rise", "jqf8qm", "ethan", "/home/", "udc-", "miniforge"]
PATHS = '''# Locations: only from the environment (run.sh sets them from config.env; see INSTRUCTIONS 2.3)
#   VM_DATA_DIR: the folder that holds RubikMemory/   VM_OUT_DIR: every run's outputs   VM_VAE: the Wan2.1 VAE folder
import os

DATA_DIR = os.environ["VM_DATA_DIR"]
OUT_DIR = os.environ["VM_OUT_DIR"]
VAE = os.environ["VM_VAE"]
'''


def main():
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(REMOTE / "package", DIST, ignore=shutil.ignore_patterns("__pycache__", ".env", ".cache", "models"))
    (DIST / "code" / "datagen").mkdir(parents=True)
    for f in CODE:
        shutil.copy2(VISUAL / f, DIST / "code" / f)
    for f in DATAGEN:
        shutil.copy2(VISUAL / "datagen" / f, DIST / "code" / "datagen" / f)
    (DIST / "code" / "paths.py").write_text(PATHS)
    problems = []
    for p in sorted(DIST.rglob("*")):
        if not p.is_file():
            continue
        s = p.read_text(encoding="utf-8")
        for a, b in SUBST:
            s = s.replace(a, b)
        p.write_text(s, encoding="utf-8")
        for bad in FORBIDDEN:
            for i, line in enumerate(s.splitlines(), 1):
                if bad in line:
                    problems.append(f"{p.relative_to(DIST)}:{i}: {bad!r} in {line.strip()[:100]}")
        if p.suffix in (".sh", ".py") and p.parent == DIST or p.name == "setup_env.sh":
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    for p in sorted((DIST / "code").rglob("*.py")):  # no top-level import of an evaluation module
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if line.startswith(("import ", "from ")) and line.split()[1].split(".")[0] in EVAL_MODULES:
                problems.append(f"{p.relative_to(DIST)}:{i}: top-level import of an evaluation module: {line.strip()}")
    env = {**os.environ, "VM_DATA_DIR": "/nonexistent", "VM_OUT_DIR": "/nonexistent", "VM_VAE": "/nonexistent",
           "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    mods = [f[:-3] for f in CODE] + [f[:-3] for f in DATAGEN]
    test = ("import sys; sys.path[:0] = ['code', 'code/datagen']; sys.modules.update({m: None for m in %r}); "
            "import %s" % (list(EVAL_MODULES), ", ".join(mods)))
    r = subprocess.run([sys.executable, "-c", test], cwd=DIST, env=env, capture_output=True, text=True)
    if r.returncode:
        problems.append("the shipped code does not import on its own:\n" + r.stderr[-1500:])
    cfg = (DIST / "config.env").read_text()
    if "VM_DATA_DIR=/path/to" not in cfg or "VM_OUT_DIR=/path/to" not in cfg:
        problems.append("config.env: the locations must be /path/to placeholders")
    if problems:
        shutil.rmtree(DIST)
        raise SystemExit("refusing to build:\n  " + "\n  ".join(problems))
    zpath = DIST.parent / f"{NAME}.zip"
    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(DIST.rglob("*")):
            if p.is_file():
                info = zipfile.ZipInfo.from_file(p, arcname=str(Path(NAME) / p.relative_to(DIST)))
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, p.read_bytes())  # keeps the permission bits (run.sh stays executable after unzip)
    files = [p for p in DIST.rglob("*") if p.is_file()]
    print(f"built {DIST} ({len(files)} files) and {zpath} ({zpath.stat().st_size / 1024:.0f} KB)")
    for p in sorted(files):
        print(f"  {p.relative_to(DIST)}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()
