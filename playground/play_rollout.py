# rollout.py 的 playground：每写完一步，这里加一段，最后停在 pdb 里随便看
#   cd /project/sds-rise/ethan/world_model && conda activate world && python playground/play_rollout.py
# 默认用 GPU 0（训练也在上面跑，测出来的时间会偏慢；这里只占几个 GB）
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 让 playground/ 里也能 import world_model/ 下的文件
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 必须在 import torch 之前设好

import torch
from torch.utils.data import default_collate

from data import ShellGameDataset
from diffusion import sigma_schedule
from model import WanModel
from rollout import Rollout, rollout

model = WanModel(causal=True).cuda().eval()
model.load_state_dict(torch.load("/project/sds-rise/ethan/Finetune/world/shellgame/ema_050000.pt", map_location="cpu"))


def load(split, n):  # 某个 split 的前 n 个 episode
    ds = ShellGameDataset("/project/sds-rise/ethan/Dataset/ShellGame", split)
    b = default_collate([ds[i] for i in range(n)])
    return b["latent"].cuda(), b["actions"].cuda()


class RefeedRollout:  # 改之前的 Rollout：不存 K/V，每一步把窗口里的干净 chunk 和当前 chunk 一起重新喂给模型
    def __init__(self, model, max_ctx=None, frames_per_chunk=5):
        self.model, self.max_ctx, self.fpc = model, max_ctx, frames_per_chunk
        self.ctx, self.ctx_a = [], []

    def commit(self, x, a):
        self.ctx.append(x)
        self.ctx_a.append(a)
        if self.max_ctx is not None:
            self.ctx, self.ctx_a = self.ctx[-self.max_ctx:], self.ctx_a[-self.max_ctx:]

    def denoise(self, x, sigma, a):
        x_all = torch.cat(self.ctx + [x], dim=2)
        a_all = torch.stack(self.ctx_a + [a], dim=1)
        t = torch.zeros(a_all.shape, device=x.device)
        t[:, -1] = sigma * 1000
        return self.model(x_all, t, a_all)[:, :, -self.fpc:]

    def sample_chunk(self, a, steps=50, shift=3.0):
        B, C, _, H, W = self.ctx[-1].shape
        sigmas = sigma_schedule(steps, shift, a.device)
        x = torch.randn(B, C, self.fpc, H, W, device=a.device)
        for i in range(steps):
            v = self.denoise(x, sigmas[i].expand(B), a)
            x = x + (sigmas[i + 1] - sigmas[i]) * v
        return x


@torch.no_grad()
def refeed_rollout(model, context, actions, steps=50, shift=3.0, max_ctx=None, frames_per_chunk=5):  # 改之前的 rollout()
    r = RefeedRollout(model, max_ctx, frames_per_chunk)
    Kc = context.shape[2] // frames_per_chunk
    for k in range(Kc):
        r.commit(context[:, :, k * frames_per_chunk:(k + 1) * frames_per_chunk], actions[:, k])
    video = [context]
    for k in range(Kc, actions.shape[1]):
        x = r.sample_chunk(actions[:, k], steps, shift)
        r.commit(x, actions[:, k])
        video.append(x)
    return torch.cat(video, dim=2)


# ---- KV-3a: 结果对不对（float32，batch 2，10 步，n10 = 13 个 chunk）----
# chunk 0–7：窗口还没开始滑，两种做法应该完全一样；chunk 8–12：窗口在滑，两种做法本来就不一样
torch.backends.cudnn.allow_tf32 = False  # 不然 Conv3d 走 TF32，两边会多出 1e-3 左右的误差
x0, a = load("test/n10", 2)
outs = []
for f in (refeed_rollout, rollout):
    torch.manual_seed(0)  # 两边的初始噪声一样
    outs.append(f(model, x0[:, :, :10], a, steps=10, max_ctx=7))
diff = (outs[0] - outs[1]).abs().amax(dim=(0, 1, 3, 4)).view(-1, 5).amax(1)  # [13]：每个 chunk 的 max |diff|
print("每个 chunk 的 max |diff|:", " ".join(f"{d:.0e}" for d in diff.tolist()))


# ---- KV-3b: 看看 Rollout 里面：commit 完 2 个 context chunk 之后 cache 里有什么 ----
r = Rollout(model, max_ctx=7)
with torch.no_grad():
    for k in range(2):
        r.commit(x0[:, :, 5 * k:5 * k + 5], a[:, k])
print(f"commit 2 个 context chunk 之后：start {r.start}，{len(r.cache)} 层，每层 {len(r.cache[0]['k'])} 个 chunk，"
      f"每个 {tuple(r.cache[0]['k'][0].shape)}")


# ---- KV-3c: 速度（bf16，batch 16 = eval 的 batch，10 步，n10）----
x0, a = load("test/n10", 16)
for f in (refeed_rollout, rollout):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        video = f(model, x0[:, :, :10], a, steps=10, max_ctx=7)
    torch.cuda.synchronize()
    print(f"{f.__name__}: {tuple(video.shape)}，{time.time() - t0:.1f} s")

breakpoint()
