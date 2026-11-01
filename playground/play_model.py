# model.py 的 playground：每写完一步，这里加一段，最后停在 pdb 里随便看
#   cd /project/sds-rise/ethan/world_model && conda activate world && python playground/play_model.py
# 默认用 GPU 0（训练也在上面跑；这个测试只占几 GB，跑几十秒，会稍微拖慢训练）
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 让 playground/ 里也能 import world_model/ 下的文件
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 必须在 import torch 之前设好

import torch
import torch.nn.functional as F

from model import chunk_attention, chunk_mask

K, T, n, d = 8, 960, 16, 72  # 8 个 chunk，每个 chunk 960 个 token，16 个头，每个头 72 维


def qkv(B, dtype):
    return [torch.randn(B, n, K * T, d, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]


def dense(q, k, v, window):  # 原来的做法：完整的 [L, L] mask
    return F.scaled_dot_product_attention(q, k, v, attn_mask=chunk_mask(K, T, window, q.device))


# ---- 6a-1: 结果一样吗（float32，差别应该只有 1e-6 左右的浮点误差）----
q, k, v = qkv(1, torch.float32)
for window in (None, 1):
    diff = (dense(q, k, v, window) - chunk_attention(q, k, v, K, window)).abs().max()
    print(f"window {window}: max |diff| {diff:.1e}")


# ---- 6a-2: 速度（bf16，前向 + 反向，batch 8 = 现在训练每张卡的 batch）----
def ms(fn, *args):
    for _ in range(3):  # 预热
        fn(*args).sum().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        fn(*args).sum().backward()
    torch.cuda.synchronize()
    return (time.time() - t0) / 10 * 1000


q, k, v = qkv(8, torch.bfloat16)
t_dense, t_chunk = ms(dense, q, k, v, None), ms(chunk_attention, q, k, v, K, None)
print(f"一层注意力 前向+反向：完整 mask {t_dense:.1f} ms | 按 chunk {t_chunk:.1f} ms | 快 {t_dense / t_chunk:.1f} 倍")


# ---- KV-1: RoPE 只看相对距离：q 在第 m 帧、k 在第 n 帧，q·k 只和 m - n 有关 ----
from torch.utils.data import default_collate

from data import ShellGameDataset
from model import WanModel, rope_apply

model = WanModel(causal=True).cuda().eval()
model.load_state_dict(torch.load("/project/sds-rise/ethan/Finetune/world/shellgame/ema_050000.pt", map_location="cpu"))

q0, k0 = torch.randn(2, 1, 192, 16, 72).unbind(0)  # 一帧的 token（12 × 16 = 192 个），16 个头，每个头 72 维


def score(m, n):  # q 放在第 m 帧，k 放在第 n 帧，算每个 token、每个头的 q·k
    q = rope_apply(q0, (1, 12, 16), model.freqs.cpu(), m)
    k = rope_apply(k0, (1, 12, 16), model.freqs.cpu(), n)
    return (q * k).sum(-1)  # [1, 192, 16]


print(f"距离都是 25：(35, 10) vs (160, 135) max |diff| {(score(35, 10) - score(160, 135)).abs().max():.1e}")
print(f"距离 25 vs 5：(35, 10) vs (35, 30) max |diff| {(score(35, 10) - score(35, 30)).abs().max():.1e}")


# ---- KV-2: 带 KV cache 的 forward 和原来一次喂 8 个 chunk 的结果一样吗（50K 的 EMA）----
# 原来：7 个干净 chunk + 1 个带噪 chunk 一起喂进去（第 0–39 帧），取最后一个 chunk 的输出
# 现在：干净 chunk 一个一个 write 进 cache，最后只喂带噪的那一个；帧号整体往后挪，结果也应该不变
torch.backends.cudnn.allow_tf32 = False  # patch_embedding 的 Conv3d 默认走 TF32，两边会差 1e-3；关掉以后应该完全一样
ds = ShellGameDataset("/project/sds-rise/ethan/Dataset/ShellGame", "test/n05")  # 5 次交换 -> 8 个 chunk
b = default_collate([ds[i] for i in range(2)])
x0, a = b["latent"].cuda(), b["actions"].cuda()  # [2, 16, 40, 24, 32], [2, 8]

sigma = 0.6
x = x0.clone()
x[:, :, 35:] = (1 - sigma) * x0[:, :, 35:] + sigma * torch.randn_like(x0[:, :, 35:])  # 只给最后一个 chunk 加噪
t = torch.zeros(2, 8, device="cuda")
t[:, -1] = sigma * 1000

with torch.no_grad():
    ref = model(x, t, a)[:, :, 35:]  # 原来的做法 [2, 16, 5, 24, 32]
    # s0 = 第 0 个 chunk 从第几帧开始；None：故意每个 chunk 都给 start = 0，看看会差多少
    for name, s0 in [("正确（第 0–39 帧）", 0), ("整体 +125（第 125–164 帧，像 N=30 最后 8 个 chunk）", 125), ("全是 0", None)]:
        cache = [{"k": [], "v": []} for _ in model.blocks]  # 每层一份
        for k in range(7):  # 前 7 个干净 chunk 一个一个存进 cache，第 k 个从第 s0 + 5k 帧开始
            model(x[:, :, 5 * k:5 * k + 5], t[:, k:k + 1], a[:, k:k + 1], cache, 0 if s0 is None else s0 + 5 * k, write=True)
        out = model(x[:, :, 35:], t[:, 7:], a[:, 7:], cache, 0 if s0 is None else s0 + 35)  # 只喂带噪的第 8 个 chunk
        print(f"start {name}: max |diff| {(out - ref).abs().max():.1e}")

breakpoint()
