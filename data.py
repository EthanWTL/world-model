import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler, Sampler


class ShellGameDataset(Dataset):
    def __init__(self, root, split="train"):
        self.dir = Path(root) / split                       # .../ShellGame/train
        with open(self.dir / "meta.jsonl") as f:
            eps = [json.loads(line) for line in f]          # 一行一个 episode -> list of dict
        self.eps = [ep for ep in eps if self.latent_path(ep).exists()]  # 只留已经编码好的

    def latent_path(self, ep):
        return self.dir / ep["video"].replace("videos/", "latents/").replace(".mp4", ".npy")

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, i):
        ep = self.eps[i]
        latent = np.load(self.latent_path(ep)) 
        return {
            "latent": torch.from_numpy(latent).float(),     # 转成 float32 给模型用
            "actions": torch.tensor(ep["actions"]),         # [K] int64，每个 chunk 一个动作
            "ball": torch.tensor(ep["ball"]),               # [K] int64，每个 chunk 结束时球在哪个杯子下
        }


def infinite_loader(ds, batch_size, num_workers=8, rank=0, world_size=1, seed=0):
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True)
    loader = DataLoader(ds, batch_size, sampler=sampler, num_workers=num_workers, drop_last=True,
                        pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    epoch = 0
    while True:
        sampler.set_epoch(epoch)  # 每个 epoch 换一种打乱顺序
        yield from loader
        epoch += 1
    #(Pdb) batch['actions'].shape
    #torch.Size([8, 8])
    #(Pdb) batch['ball'].shape
    #torch.Size([8, 8])
    #(Pdb) batch['latent'].shape
    #torch.Size([8, 16, 40, 24, 32])


class CurriculumSampler(Sampler):
    # 课程：一个 batch 里只有一种交换次数 N（chunk 数一样才能拼成一个 batch）
    # 每步先均匀抽一个 N，再从这个 N 的 episode 里有放回地抽 batch_size 个；无限循环
    def __init__(self, buckets, batch_size, seed=0):
        self.buckets = [torch.tensor(b) for b in buckets]  # buckets[j]：交换次数是第 j 个 N 的 episode 下标
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)

    def __iter__(self):
        while True:
            b = self.buckets[torch.randint(len(self.buckets), (1,), generator=self.g).item()]
            yield b[torch.randint(len(b), (self.batch_size,), generator=self.g)].tolist()


def curriculum_loader(batch_size, num_workers=8, seed=0, ns=range(6), root="/project/sds-rise/ethan/Dataset/ShellGame",
                      curr_root="/project/sds-rise/ethan/Dataset/ShellGameCurr"):
    # N = 5 来自 ShellGame/train（30000 个）；N = 0..4 来自 ShellGameCurr/train（5 / 50 / 500 / 4000 / 4000 个）
    # 交换 N 次的 episode 有 N + 3 个 chunk：揭开、盖上、N 次交换、最后揭开
    main, curr = ShellGameDataset(root, "train"), ShellGameDataset(curr_root, "train")
    eps = main.eps + curr.eps  # 和 ConcatDataset 的下标一一对应
    buckets = [[i for i, ep in enumerate(eps) if len(ep["actions"]) - 3 == n] for n in ns]
    assert all(buckets), [len(b) for b in buckets]
    loader = DataLoader(ConcatDataset([main, curr]), batch_sampler=CurriculumSampler(buckets, batch_size, seed),
                        num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    yield from loader
