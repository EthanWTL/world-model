# latent 文件的位置、Wan2.1 VAE 的加载和编码、多 worker 抢 shard（两个任务的造数据都用；远端包也带它）
import os
import socket
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import VAE  # noqa: E402


def latent_path(root, split, i):
    return Path(root) / split / "latents" / f"{i // 1000:02d}" / f"{i:06d}.npy"


def load_vae():
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(VAE, subfolder="vae",
                                           torch_dtype=torch.float32).cuda().eval()
    mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda").view(1, 16, 1, 1, 1)
    return vae, mean, std


@torch.no_grad()
def encode(vae, mean, std, frames):
    # frames: list of uint8 [T, H, W, 3]（同样长）-> float16 [B, 16, (T + 3) / 4, 24, 32]
    x = torch.from_numpy(np.stack(frames)).cuda().permute(0, 4, 1, 2, 3).float() / 127.5 - 1
    return ((vae.encode(x).latent_dist.mode() - mean) / std).half()


def claim(root, shard, num_shards):
    d = Path(root) / "logs" / "claims"
    d.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(d / f"{shard:03d}of{num_shards}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{socket.gethostname()} job {os.environ.get('SLURM_JOB_ID', '-')} pid {os.getpid()}\n".encode())
    os.close(fd)
    return True
