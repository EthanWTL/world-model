# 推理（远端包的 infer 阶段，本地也能用）：一个 checkpoint 在魔方测试集上按两种协议展开，只存生成的 latent，不算任何指标
#   free：给 chunk 0–3（不动 + 开头展示），之后全部自己生成 -> <out>/nXX/free.npy  float16 [E, 16, 5K, 24, 32]（整段；前 4 个 chunk = GT）
#   tf  ：除了最后展示的 3 个 chunk 都给 GT，只生成最后展示  -> <out>/nXX/tf.npy    float16 [E, 16, 20, 24, 32]（最后 4 个 chunk；
#         第一个是 GT：打乱的最后一个 chunk）
#   <out>/nXX/ids.json：episode id 的顺序；<out>/done.json：设置（ckpt、step、max_ctx、采样步数、种子、形状、用时）
#   噪声：每个 episode 自己的 generator（种子 = seed·100000 + episode id），和本地评测（rubik_eval.py）一样 -> 同一个 ckpt 对得上
#   评测在本地：python rubik_eval.py --saved <out> --out <json>
#   python infer.py --ckpt <OUT>/rk_qkv_norm/ckpt_012000.pt --out <OUT>/inference/rk_qkv_norm/ckpt_012000
#   python infer.py --ckpt <OUT>/rubik_base/ema_040000.pt --max-ctx -1 --out <OUT>/inference/rubik_base/ema_040000_ctx-1
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import default_collate

from data import TASKS, CubeDataset
from sampling import build_from_ckpt, rollout


def save(path, arr):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--key", default=None)                      # ckpt 里用 "ema" 还是 "model"（默认 ema）
    ap.add_argument("--max-ctx", type=int, default=-2)          # -2 = ckpt 里的设置，-1 = KV cache 全留，0 = 一个都不留
    ap.add_argument("--out", required=True)
    ap.add_argument("--moves", type=int, nargs="+", default=list(TASKS["rubik"]["test"]))
    ap.add_argument("--protocols", nargs="+", default=["free", "tf"], choices=["free", "tf"])
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, cargs, step = build_from_ckpt(args.ckpt, args.key, args.max_ctx)
    model.eval()
    t, t0, shapes = TASKS["rubik"], time.time(), {}
    print(f"infer {args.ckpt} ({args.key or 'ema'}, step {step}) | max_ctx {model.max_ctx} | moves {args.moves} | "
          f"{args.protocols} | {args.steps} steps | seed {args.seed} -> {out}", flush=True)
    for n in args.moves:
        split = t["split"].format(n)
        d = out / split.split("/")[-1]
        if all((d / f"{p}.npy").exists() for p in args.protocols):
            print(f"{split}: done already", flush=True)
            continue
        ds = CubeDataset(split, t["root"])
        E = min(args.num, len(ds))
        if E == 0:
            raise SystemExit(f"{split}: no episodes with latents (data stage incomplete?)")
        K = ds.eps[0]["num_chunks"]
        bs = args.batch if model.max_ctx is not None else max(1, args.batch * 12 // K)  # KV 全留：长 episode 缩 batch
        res = {p: [] for p in args.protocols}
        for i in range(0, E, bs):
            idx = list(range(i, min(i + bs, E)))
            b = default_collate([ds[j] for j in idx])
            x0, a = b["latent"].cuda(), b["actions"].cuda()
            seeds = [args.seed * 100000 + ds.eps[j]["id"] for j in idx]
            for p in args.protocols:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, _, z = rollout(model, x0, a, K - 3 if p == "tf" else 4, args.steps, seeds, keep=True)
                res[p].append((z if p == "free" else z[:, :, 5 * (K - 4):]).half().cpu().numpy())
        d.mkdir(parents=True, exist_ok=True)
        (d / "ids.json").write_text(json.dumps([ds.eps[j]["id"] for j in range(E)]))
        for p, zs in res.items():
            arr = np.concatenate(zs)
            save(d / f"{p}.npy", arr)
            shapes[f"{split}/{p}"] = list(arr.shape)
        print(f"{split}: {E} episodes, K {K}, {', '.join(args.protocols)} saved ({time.time() - t0:.0f} s)", flush=True)
    (out / "done.json").write_text(json.dumps({
        "ckpt": args.ckpt, "key": args.key or "ema", "step": step, "max_ctx": -1 if model.max_ctx is None else model.max_ctx,
        "run_args": {k: cargs.get(k) for k in ("run", "task", "size", "mem_proj", "mem_ffn", "mem_key", "mem_loss",
                                                "mem_fast_heads", "mem_blocks", "steps") if k in cargs},
        "moves": args.moves, "protocols": args.protocols, "num": args.num, "sampling_steps": args.steps, "seed": args.seed,
        "shapes": shapes, "seconds": round(time.time() - t0), "time": time.strftime("%F %T")}, indent=1))
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
