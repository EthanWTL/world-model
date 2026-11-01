# data.py 的 playground：每写完一步，这里加一段，最后停在 pdb 里随便看
#   cd /project/sds-rise/ethan/world_model && conda activate world && python playground/play_data.py
# pdb 常用：p x.shape | pp ep | interact（进完整 Python，Ctrl-D 回 pdb）| c 继续 | q 退出
# 想单步进自己写的函数：在 data.py 那一行加 breakpoint()，再跑这个脚本，会停在函数里面
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 测数据不用 GPU，别占共享机器的卡
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 让 playground/ 里也能 import world_model/ 下的文件

from torch.utils.data import DistributedSampler, default_collate

from data import ShellGameDataset, infinite_loader

ROOT = "/project/sds-rise/ethan/Dataset/ShellGame"

# ---- 1a: 索引（__init__ + latent_path）----
ds = ShellGameDataset(ROOT, "train")
print(f"train: {len(ds.eps)} 个 episode 有 latent")
ep = ds.eps[0]
print("  eps[0]:", ep)
print("  latent:", ds.latent_path(ep))

# ---- 1b: 取样本（__len__ + __getitem__）----
sample = ds[0]
print("sample[0]:")
for k, v in sample.items():
    print(f"  {k:8s} {tuple(v.shape)} {v.dtype}")
z = sample["latent"]
print(f"  latent mean {z.mean():.3f} std {z.std():.3f} | actions {sample['actions'].tolist()} | ball {sample['ball'].tolist()}")

test = ShellGameDataset(ROOT, "test/n30")  # 30 次交换 -> 33 个 chunk
print(f"test/n30: {len(test)} 个 episode, latent {tuple(test[0]['latent'].shape)}, actions {tuple(test[0]['actions'].shape)}")

batch = default_collate([ds[i] for i in range(4)])  # DataLoader(batch_size=4) 就是这样把 4 个样本拼成一个 batch
print("batch:", {k: tuple(v.shape) for k, v in batch.items()})

# ---- 2: infinite_loader ----
s0 = DistributedSampler(ds, num_replicas=2, rank=0, shuffle=True, seed=0, drop_last=True)  # 假装有 2 张卡
s1 = DistributedSampler(ds, num_replicas=2, rank=1, shuffle=True, seed=0, drop_last=True)
print("sampler: rank0 前 6 个 index", list(s0)[:6], "| rank1", list(s1)[:6])
s0.set_epoch(1)
print("         rank0 换到 epoch 1 后", list(s0)[:6])

batches = infinite_loader(ds, batch_size=8, num_workers=4)
t0 = time.time()
for _ in range(20):
    batch = next(batches)
print(f"loader: batch latent {tuple(batch['latent'].shape)}, actions {tuple(batch['actions'].shape)}, "
      f"{(time.time() - t0) / 20 * 1000:.0f} ms/batch（20 个 batch 的平均，含 worker 启动）")

breakpoint()
