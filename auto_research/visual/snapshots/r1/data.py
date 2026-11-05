# 旋转方块的 latent 数据（datagen/build.py 造的 Dataset/CubeMemory）
#   train：gap ~ U{0, 2, 4, 6}；test/gXX：每个 gap 64 个，所有配置共用
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = "/project/sds-rise/ethan/Dataset/CubeMemory"


class CubeDataset(Dataset):
    def __init__(self, split="train", root=ROOT):
        self.dir = Path(root) / split
        with open(self.dir / "meta.jsonl") as f:
            eps = [json.loads(line) for line in f]
        have, lat = set(), self.dir / "latents"  # 列目录比一个个 stat 快得多（6 万个文件）；.tmp 半成品不算
        if lat.exists():
            for sub in os.listdir(lat):
                have.update(f"{sub}/{f}" for f in os.listdir(lat / sub) if f.endswith(".npy"))
        self.eps = [ep for ep in eps if f"{ep['id'] // 1000:02d}/{ep['id']:06d}.npy" in have]

    def latent_path(self, ep):
        return self.dir / "latents" / f"{ep['id'] // 1000:02d}" / f"{ep['id']:06d}.npy"

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, i):
        ep = self.eps[i]
        return {"latent": torch.from_numpy(np.load(self.latent_path(ep))).float(),  # [16, 5K, 24, 32]
                "actions": torch.tensor(ep["actions"]),                            # [K]
                "idx": i}


class GapSampler(torch.utils.data.Sampler):
    # 每步按 probs 抽一个 gap（一个 batch 里 gap 一样 = chunk 数一样），再在这个桶里有放回地抽 batch 个
    # seed_gap 只管抽 gap：DDP 时每张卡给一样的 seed_gap，同一步大家的序列一样长；seed 管抽 episode（每张卡不同）
    def __init__(self, buckets, probs, batch, seed=0, seed_gap=None):
        self.buckets = [torch.tensor(b) for b in buckets]
        self.probs = torch.tensor(probs, dtype=torch.float)
        self.batch = batch
        self.g = torch.Generator().manual_seed(seed)
        self.gg = torch.Generator().manual_seed(seed if seed_gap is None else seed_gap)

    def __iter__(self):
        while True:
            b = self.buckets[torch.multinomial(self.probs, 1, generator=self.gg).item()]
            yield b[torch.randint(len(b), (self.batch,), generator=self.g)].tolist()


def train_loader(batch, gaps=(0, 2, 4, 6), probs=None, num_workers=6, seed=0, seed_gap=None, root=ROOT):
    ds = CubeDataset("train", root)
    buckets = [[i for i, ep in enumerate(ds.eps) if ep["gap"] == g] for g in gaps]
    assert all(buckets), [len(b) for b in buckets]
    sampler = GapSampler(buckets, probs or [1.0] * len(gaps), batch, seed, seed_gap)
    yield from DataLoader(ds, batch_sampler=sampler, num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0)
