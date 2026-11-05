#!/usr/bin/env bash
# The one environment of the package, installed in this order:
#   1. python 3.11 (conda-forge)
#   2. torch 2.8.0 (PyTorch index; TORCH_CUDA=cu128 by default: CUDA 12.8 runtime bundled, kernels for A100 / H100 /
#      H200 (sm_80 / sm_90) and Blackwell (sm_100 / sm_120); NVIDIA driver >= 570 recommended, >= 525 works through
#      CUDA minor-version compatibility; TORCH_CUDA=cu126 for older drivers, no Blackwell)
#   3. requirements.lock.txt: every other package at its exact version (requirements.txt = the direct ones)
#
#   ./setup_env.sh                                   # -> ./.env next to this script; needs conda (or CONDA=mamba) on PATH
#   ENV_PREFIX=/path/to/envs/rubik_memory ./setup_env.sh    # elsewhere (then PYTHON=/path/to/envs/rubik_memory/bin/python ./run.sh ...)
#
# Nothing is written outside this folder and ENV_PREFIX: conda's package cache, its home files and temp files go to
# ./.cache/ (deletable after the setup); pip runs without a cache. conda runs with HOME=./.cache/home and the
# conda-forge channel only; KEEP_HOME=1 keeps the real HOME (e.g. for proxy settings in ~/.condarc).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$HERE/.env}"
CONDA="${CONDA:-conda}"
CACHE="${CACHE_DIR:-$HERE/.cache}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
case "$TORCH_CUDA" in cu126|cu128) ;; *) echo "TORCH_CUDA must be cu126 or cu128"; exit 1 ;; esac

export CONDA_PKGS_DIRS="$CACHE/conda_pkgs" CONDA_REGISTER_ENVS=false PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export TMPDIR="$CACHE/tmp" XDG_CACHE_HOME="$CACHE/xdg/cache" PYTHONNOUSERSITE=1
mkdir -p "$CONDA_PKGS_DIRS" "$TMPDIR" "$XDG_CACHE_HOME" "$CACHE/home"

echo "== 1/3 python 3.11 -> $ENV_PREFIX"
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  if [ "${KEEP_HOME:-0}" = 1 ]; then
    "$CONDA" create -y -p "$ENV_PREFIX" -c conda-forge --override-channels "python=3.11" pip
  else
    HOME="$CACHE/home" "$CONDA" create -y -p "$ENV_PREFIX" -c conda-forge --override-channels "python=3.11" pip
  fi
fi
PY="$ENV_PREFIX/bin/python"
"$PY" -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"

echo "== 2/3 torch 2.8.0+$TORCH_CUDA"
"$PY" -m pip install "torch==2.8.0+$TORCH_CUDA" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

echo "== 3/3 the rest (requirements.lock.txt, exact versions)"
"$PY" -m pip install -r "$HERE/requirements.lock.txt"

echo "== check"
"$PY" - "$TORCH_CUDA" "$HERE/requirements.lock.txt" <<'PYEOF'
import importlib.metadata as m, sys
want = {"torch": f"2.8.0+{sys.argv[1]}"}
for line in open(sys.argv[2]):
    if "==" in line and not line.startswith("#"):
        name, ver = line.strip().split("==")
        want[name] = ver
bad = {k: (m.version(k), v) for k, v in want.items() if m.version(k) != v}
print("python", sys.version.split()[0], "| torch", m.version("torch"), "| diffusers", m.version("diffusers"), f"| {len(want)} pinned packages")
assert not bad and sys.version_info[:2] == (3, 11), f"versions differ (installed, wanted): {bad}"
import diffusers, imageio_ffmpeg, torch  # noqa: F401
PYEOF
echo "env ready: $ENV_PREFIX"
echo "(the setup caches in $CACHE can be deleted now)"
