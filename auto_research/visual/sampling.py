# 采样：σ 表、按 episode 定的噪声、逐 chunk 的 Euler 采样、整段展开（前 n_ctx 个 chunk 给 GT，之后自己生成）、从 ckpt 建模型
# 评测（evaluate.py / rubik_eval.py）和推理（infer.py）共用；远端包带它（不带评测）
import torch

from data import TASKS
from wm import WanMem, commit, mem_config


def sigma_schedule(steps=50, shift=3.0, device="cuda"):
    u = torch.linspace(1, 0, steps + 1, device=device)
    return shift * u / (1 + (shift - 1) * u)


def randn(shape, gens):
    # 每个样本用自己的 generator（种子 = episode 编号）：换 batch 大小也是同一份噪声，所有配置共用
    return torch.stack([torch.randn(shape[1:], device="cuda", generator=g) for g in gens])


@torch.no_grad()
def sample_chunk(model, st, a, shape, steps, gens):
    sig = sigma_schedule(steps)
    x = randn(shape, gens)
    for i in range(steps):
        v, _ = model(x, sig[i].expand(shape[0]) * 1000, a, st)
        x = x + (sig[i + 1] - sig[i]) * v.float()
    return x


@torch.no_grad()
def rollout(model, x0, a, n_ctx, steps, seeds, keep=False):
    # 前 n_ctx 个 chunk 用 GT，之后每个 chunk 先生成再 commit（写记忆）；最后一个 chunk 只生成
    # seeds：每个样本一个；返回 (最后一个 chunk 的采样, σ=1 一步预测的 x0, 整段 latent（keep 时，否则 None）)
    gen = [torch.Generator(device="cuda").manual_seed(s) for s in seeds]
    B, K = a.shape
    shape = (B, 16, 5, x0.shape[3], x0.shape[4])
    st, chunks = model.init_state(), []
    for c in range(K - 1):
        x = x0[:, :, 5 * c:5 * c + 5] if c < n_ctx else sample_chunk(model, st, a[:, c], shape, steps, gen)
        if keep:
            chunks.append(x)
        st = commit(model, x, a[:, c], st)
    noise = randn(shape, gen)
    v, _ = model(noise, torch.full((B,), 1000.0, device="cuda"), a[:, -1], st)
    x0_hat = noise - v.float()  # σ=1：x_t = 噪声，x0 = x_t - σ·v
    x = sample_chunk(model, st, a[:, -1], shape, steps, gen)
    return x, x0_hat, (torch.cat(chunks + [x], 2) if keep else None)


def build_from_ckpt(path, key=None, max_ctx=-2):
    # train.py 的 ckpt（带记忆）或 train_base.py 的 ema_*.pt（没有记忆）；max_ctx：-2 = 用 ckpt 里的，-1 = 全留
    # 任务（动作数）看 ckpt 的 args["task"]，没有 = 旋转方块
    ck = torch.load(path, map_location="cpu")
    args = ck["args"]
    na = TASKS[args.get("task", "cube")]["num_actions"]
    if "mem_proj" in args:
        model = WanMem(args.get("size", "200m"), mem_config(args), max_ctx=args["max_ctx"], num_actions=na)
    else:
        model = WanMem(args.get("size", "200m"), max_ctx=None, num_actions=na)
    model.load_state_dict(ck[key] if key else (ck["ema"] if "ema" in ck else ck["model"]))
    if max_ctx != -2:
        model.max_ctx = None if max_ctx == -1 else max_ctx
    return model.cuda(), args, ck.get("step")
