import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


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
