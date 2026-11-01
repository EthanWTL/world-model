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


class CurriculumSampler(torch.utils.data.Sampler):
    # 每个 batch 里交换次数 N 一样（chunk 数一样才能拼成一个 batch）：每步按 probs 抽一个 N，再在这个桶里有放回地抽 batch 个
    def __init__(self, buckets, probs, batch, seed=0):
        self.buckets = [torch.tensor(b) for b in buckets]
        self.probs = torch.tensor(probs, dtype=torch.float)
        self.batch = batch
        self.g = torch.Generator().manual_seed(seed)

    def __iter__(self):
        while True:
            b = self.buckets[torch.multinomial(self.probs, 1, generator=self.g).item()]
            yield b[torch.randint(len(b), (self.batch,), generator=self.g)].tolist()


def curriculum_loader(batch_size, num_workers=6, seed=0, ns=(0, 1, 2, 3, 4, 5),
                      curr_root="/project/sds-rise/ethan/Dataset/ShellGameCurr"):
    # N=5 来自主训练集，N=0..4 来自 ShellGameCurr；每步均匀抽一个 N
    main, curr = ShellGameDataset(split="train"), ShellGameDataset(curr_root, "train")
    ds = torch.utils.data.ConcatDataset([main, curr])
    eps = main.eps + curr.eps
    buckets = [[i for i, ep in enumerate(eps) if len(ep["actions"]) - 3 == n] for n in ns]
    assert all(buckets), [len(b) for b in buckets]
    sampler = CurriculumSampler(buckets, [1.0] * len(ns), batch_size, seed)
    yield from DataLoader(ds, batch_sampler=sampler, num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0)
