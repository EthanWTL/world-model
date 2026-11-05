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


# ---- 多球（Dataset/ShellGameMulti，datagen/gen_multi.py）----
MULTI = "/project/sds-rise/ethan/Dataset/ShellGameMulti"


class MultiBallDataset(ShellGameDataset):
    # split = "k3/train"、"k3/test/n05"…；再返回 balls [K, 5]：每个 chunk 结束时每个球在几号位（没有的球 = -1）
    # （ball 还是球 0 = 红球的位置，probe 照旧用它）
    def __getitem__(self, i):
        d = super().__getitem__(i)
        ep = self.eps[i]
        balls = torch.full((len(ep["actions"]), 5), -1)
        balls[:, :ep["k"]] = torch.tensor(ep["balls"])
        d["balls"] = balls
        return d


class MultiSampler(torch.utils.data.Sampler):
    # 每步抽一个 N（一个 batch 的 N 一样，chunk 数才一样），batch 里每个样本再各抽一个 k，从 (k, N) 桶里有放回地抽
    # seed_n 只管抽 N：DDP 时每张卡给一样的 seed_n，同一步大家的序列一样长；seed 管抽 k 和 episode（每张卡不同）
    def __init__(self, buckets, ks, ns, batch, seed=0, seed_n=None):
        self.buckets = {key: torch.tensor(v) for key, v in buckets.items()}
        self.ks, self.ns, self.batch = list(ks), list(ns), batch
        self.g = torch.Generator().manual_seed(seed)
        self.gn = torch.Generator().manual_seed(seed if seed_n is None else seed_n)

    def __iter__(self):
        while True:
            n = self.ns[torch.randint(len(self.ns), (1,), generator=self.gn).item()]
            out = []
            for j in torch.randint(len(self.ks), (self.batch,), generator=self.g).tolist():
                b = self.buckets[(self.ks[j], n)]
                out.append(b[torch.randint(len(b), (1,), generator=self.g)].item())
            yield out


def multi_curriculum_loader(batch_size, ks, num_workers=6, seed=0, seed_n=None, ns=(0, 1, 2, 3, 4, 5), root=MULTI):
    # ks = [3]：只用 3 个球的数据（TTT 一个 run 一个 k）；ks = [1, 2, 3, 4, 5]：混在一起（train_base.py）
    dss = [MultiBallDataset(root, f"k{k}/train") for k in ks]
    eps = [ep for ds in dss for ep in ds.eps]  # 和 ConcatDataset 的下标一一对应
    buckets = {(k, n): [i for i, ep in enumerate(eps) if ep["k"] == k and len(ep["actions"]) - 3 == n] for k in ks for n in ns}
    assert all(buckets.values()), {key: len(v) for key, v in buckets.items()}
    sampler = MultiSampler(buckets, ks, ns, batch_size, seed, seed_n)
    yield from DataLoader(torch.utils.data.ConcatDataset(dss), batch_sampler=sampler, num_workers=num_workers,
                          pin_memory=True, persistent_workers=num_workers > 0)
