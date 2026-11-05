#!/usr/bin/env bash
# Downloads the one external model file set: the Wan2.1 VAE (diffusers layout, the vae/ folder only, 0.5 GB) from the
# public Hugging Face repo Wan-AI/Wan2.1-T2V-1.3B-Diffusers at a pinned commit, and checks both files' sha256.
#
#   ./download.sh                                   # -> ./models/Wan2.1-T2V-1.3B-Diffusers (the default VM_VAE of run.sh)
#   VAE_DIR=/path/to/models/Wan2.1-T2V-1.3B-Diffusers ./download.sh   # elsewhere; then export VM_VAE=<that folder>
#
# Needs internet (setup only). Writes nothing outside VAE_DIR and ./.cache/download (Hugging Face cache, temp files).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAE_DIR="${VAE_DIR:-$HERE/models/Wan2.1-T2V-1.3B-Diffusers}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$HERE/env/.env/bin/python" ]]; then PYTHON="$HERE/env/.env/bin/python"; else PYTHON=python; fi
fi
mkdir -p "$VAE_DIR" "$HERE/.cache/download/hf" "$HERE/.cache/download/tmp"
export HF_HOME="$HERE/.cache/download/hf" TMPDIR="$HERE/.cache/download/tmp" HF_HUB_DISABLE_TELEMETRY=1 PYTHONNOUSERSITE=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
"$PYTHON" - "$VAE_DIR" <<'EOF'
import hashlib, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
repo, rev, out = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "0fad780a534b6463e45facd96134c9f345acfa5b", Path(sys.argv[1])
want = {"vae/config.json": (724, "f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527"),
        "vae/diffusion_pytorch_model.safetensors": (507591892, "d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793")}
for name, (size, sha) in want.items():
    p = Path(hf_hub_download(repo, name, revision=rev, local_dir=out))
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    ok = p.stat().st_size == size and h.hexdigest() == sha
    print(f"{name}: {p.stat().st_size} bytes, sha256 {'OK' if ok else 'MISMATCH ' + h.hexdigest()}")
    assert ok, f"{name} differs from the pinned file"
print(f"VAE ready: {out} (run.sh uses it as VM_VAE)")
EOF
