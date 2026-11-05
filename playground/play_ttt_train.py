# ttt_train.py 的 playground：默认配置走一遍训练的一个 step（data -> unroll -> loss -> backward -> opt.step），中间自己打 breakpoint() 看
#   cd /project/sds-rise/ethan/world_model && conda activate world
#   CUDA_VISIBLE_DEVICES=0 python playground/play_ttt_train.py          base 的权重（= 训练第 1 步）：记忆零初始化，写完 ΔW 还是 0
#   CUDA_VISIBLE_DEVICES=0 python playground/play_ttt_train.py --ckpt /project/sds-rise/ethan/Finetune/world/ttt/a1_ref_s1/ckpt_008000.pt
#                                                                      训好的（默认配置、已经学会的 run）：ΔW 真的写进了东西
#   --n：这个 batch 交换几次（默认 1 -> 4 个 chunk；正式训练每步随机 0..5）；其余参数和 ttt_train.py 一样（--mem-proj k、--batch 8 ...）
# 和正式训练只差三个参数，都是为了好打断点：
#   --batch 2；--workers 0（DataLoader 子进程里的 breakpoint 停不下来）
#   --grad-ckpt 0（开着的话 backward 会把每个 block 重算一遍，block_chunk / read / write 里的断点会在 backward 时再停一轮）
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 让 playground/ 里也能 import world_model/ 下的文件
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 必须在 import torch 之前设好

import torch
import torch.nn.functional as F

from data import curriculum_loader
from ttt_train import build, get_args, mem_stats, unroll, update_ema

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None)
ap.add_argument("--n", type=int, default=1)
pa, rest = ap.parse_known_args()
args = get_args(["--run", "play", "--batch", "2", "--workers", "0", "--grad-ckpt", "0", *rest])  # 命令行给的会覆盖前面的
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---- 1: data：课程 loader 拿一个 batch（ttt_train.py:147 -> data.py:63）----
# 一个 batch 里交换次数都一样（chunk 数一样才拼得起来）；交换 N 次 = N + 3 个 chunk：揭开、盖上、N 次交换、揭开
# 动作：0..9 交换两个杯子、10 揭开、11 盖上；ball：每个 chunk 结束时球在 5 个位置里的哪个
# 预期：latent (2, 16, 20, 24, 32) float32（4 个 chunk × 5 个 latent 帧）| actions [[10, 11, 5, 10], [10, 11, 6, 10]]
#       ball [[2, 2, 2, 2], [4, 4, 1, 1]]：交换 5 = 杯子 (1, 3)，球在 2 不动；交换 6 = (1, 4)，球从 4 换到 1
batches = curriculum_loader(args.batch, args.workers, seed=args.seed, ns=(pa.n,))
b = next(batches)
x0, a = b["latent"].cuda(), b["actions"].cuda()

B, K = a.shape
print(f"N {pa.n} -> K {K} 个 chunk | latent {tuple(x0.shape)} {x0.dtype} | actions {a.tolist()} | ball {b['ball'].tolist()}")


# ---- 2: 模型 + 优化器（ttt_train.py:84 build）+ 空的 state ----
# 预期：新参数 4.00M（12 层 × q 的记忆：target 1152->288 + eta 1152->1）
#       优化器三组：NorMuon 89 个矩阵（base 的 77 个 + 12 个 target）| AdamW 117 个 | η 24 个（12 层 eta 的 weight + bias，lr 1e-3）
#       init_state：kv 12 个空 list | mem[0] {'q': (2, 288, 1152)} 全 0 | start 0
model, ema, opt = build(args)

if pa.ckpt:  # 和续训一样：model / ema 都换成训好的（参数要和 ckpt 对上，默认配置的 run：a1_ref_s0 / a1_ref_s1）
    ck = torch.load(pa.ckpt, map_location="cuda")
    model.load_state_dict(ck["model"])
    ema.load_state_dict(ck["ema"])
n_new = sum(p.numel() for n, p in model.named_parameters() if n.startswith("mems."))
print(f"新参数 {n_new / 1e6:.2f}M | 优化器", [(len(g["params"]), "NorMuon" if g.get("muon") else "AdamW", g["lr"])
                                             for g in opt.param_groups])
st0 = model.init_state(B)
print("init_state: kv", st0.kv, "| mem[0]", {p: tuple(s.shape) for p, s in st0.mem[0].items()}, "| start", st0.start)

# ---- 3: 一个训练 step（和 ttt_train.py:177-184 的循环体一样，只是没有 warmup）----
# 预期（base）：L (2, 3) 每个 chunk [0.1979, 0.6196, 0.3184] | loss 0.3786 | start 15 | 窗口 0 | dW_q 0（target 零初始化，写进去的都是 0）
#              梯度 target 2.2e-04 | eta 严格是 0（什么都没写，η 多大都一样）| q 的 slow weight 1.2e-02
#     （--ckpt）：每个 chunk [0.0001, 0.004, 0.0002] | loss 0.0014 | dW_q 1.62e-01（ΔW 是 W_fast 的 16%）| eta 梯度 1.8e-05
with torch.autocast("cuda", dtype=torch.bfloat16):# 起点：s 进 unroll 一行一行跟；或者 c 直接跑到你在 ttt_model.py / memory.py 里打的断点
    L, st = unroll(model, x0, a)  # L: [B, K-1]，每个样本每个 denoise chunk 一个 loss；st：commit 完 K-1 个 chunk 的 state
loss = L.mean()
opt.zero_grad(set_to_none=True)
loss.backward()  # 穿过所有 commit 写进去的 ΔW 反传（BPTT）
gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
update_ema(ema, model, args.ema)
print(f"L {tuple(L.shape)} 每个 chunk {[round(v, 4) for v in L.mean(0).tolist()]} | loss {loss.item():.4f} | grad norm {gn.item():.3f}")
print("state: start", st.start, "| 窗口", len(st.kv[0]), "| ΔW", tuple(st.mem[0]["q"].shape), st.mem[0]["q"].dtype,
      "|", {k: f"{v:.2e}" for k, v in mem_stats(model, st).items()})
print("梯度 层 0：target", f"{model.mems[0]['q'].target.weight.grad.norm().item():.1e}",
      "| eta", f"{model.mems[0]['q'].eta.weight.grad.norm().item():.1e}",
      "| q 的 slow weight", f"{model.blocks[0].self_attn.q.weight.grad.norm().item():.1e}")

breakpoint()
