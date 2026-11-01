# Shell Game latent 数据（和 world_model/data.py 一样读 Dataset/ShellGame/<split>/latents）
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = "/project/sds-rise/ethan/Dataset/ShellGame"


class ShellGameDataset(Dataset):
    def __init__(self, root=ROOT, split="train"):
        self.dir = Path(root) / split
        with open(self.dir / "meta.jsonl") as f:
            eps = [json.loads(line) for line in f]
        self.eps = [ep for ep in eps if self.latent_path(ep).exists()]

    def latent_path(self, ep):
        return self.dir / ep["video"].replace("videos/", "latents/").replace(".mp4", ".npy")

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, i):
        ep = self.eps[i]
        return {"latent": torch.from_numpy(np.load(self.latent_path(ep))).float(),  # [16, 5K, 24, 32]
                "actions": torch.tensor(ep["actions"]),                            # [K]
                "ball": torch.tensor(ep["ball"])}                                  # [K] 每个 chunk 结束时球在几号位


def infinite_loader(ds, batch_size, num_workers=6, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(ds, batch_size, shuffle=True, num_workers=num_workers, drop_last=True, generator=g,
                        pin_memory=True, persistent_workers=num_workers > 0)
    while True:
        yield from loader
