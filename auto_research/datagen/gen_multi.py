# 多球 Shell Game（auto_research）：k = 1..5 个不同颜色的球，各在不同的杯子下面；其余（5 个杯子、动作、chunk 划分、渲染）和 ShellGame 一样。
# 球 b 的颜色固定是 PALETTE[b]（红、绿、蓝、黄、紫，见 render_multi.html），所以 k=1 就是原来的单个红球。
# 标签：balls[c][b] = 第 c 个 chunk 结束时球 b 在几号位；ball / start / final 是球 0（红）的，和单球数据的格式兼容。
#   Dataset/ShellGameMulti/k{K}/train       N = 0..5 的课程数据，每个 N 的个数 = min(--counts[N], 不重复的上限 P(5,K)·10^N)
#   Dataset/ShellGameMulti/k{K}/test/nXX    N = 5 / 10 / 20 / 30 各 64 个，和 train 不重复
#   python gen_multi.py --debug 5 0 4 7     # 5 个球（依次放在 0..4 号位）、交换动作 0 4 7 -> <out>/debug_k5.mp4
#   sbatch render_multi.slurm               # CPU 数组；重跑会跳过已经渲染好的
import argparse
import json
import math
import shlex
from pathlib import Path

import numpy as np

from generate import LIFT, LOWER, PAIRS, find_chrome, render

PAGE = "render_multi.html"


def episode(starts, swaps):
    balls = [list(starts), list(starts)]  # 揭开、盖上：还没换
    for a in swaps:
        i, j = PAIRS[a]
        balls.append([j if s == i else i if s == j else s for s in balls[-1]])
    balls.append(balls[-1])  # 最后揭开
    return {"k": len(starts), "starts": list(starts), "swaps": [list(PAIRS[a]) for a in swaps],
            "actions": [LIFT, LOWER, *swaps, LIFT], "balls": balls, "finals": balls[-1],
            "ball": [b[0] for b in balls], "start": starts[0], "final": balls[-1][0], "num_frames": 20 * len(swaps) + 57}


def sample(rng, k, n, count, taken):
    # 不重复的 (起始位置, 交换序列)；taken 在同一个 k 的所有 split 之间共享，所以 test 不会和 train 重复
    count = min(count, math.perm(5, k) * 10 ** n)
    out = []
    while len(out) < count:
        ep = (tuple(rng.permutation(5)[:k].tolist()), tuple(rng.integers(10, size=n).tolist()))
        if ep not in taken:
            taken.add(ep)
            out.append(ep)
    return out


def build(k, counts, test_ns, test_num, seed):
    rng = np.random.default_rng([seed, k])
    taken = set()
    splits = {f"k{k}/train": [episode(*e) for n, c in enumerate(counts) for e in sample(rng, k, n, c, taken)]}
    for n in test_ns:
        splits[f"k{k}/test/n{n:02d}"] = [episode(*e) for e in sample(rng, k, n, test_num, taken)]
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/project/sds-rise/ethan/Dataset/ShellGameMulti")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--counts", type=int, nargs="+", default=[120, 1000, 3000, 3000, 3000, 3000])  # N = 0..5
    ap.add_argument("--test-ns", type=int, nargs="+", default=[5, 10, 20, 30])
    ap.add_argument("--test-num", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--sub", type=int, default=0)  # 再把一个 shard 拆成 num-subs 份（小任务好排队），只有 shard 0 写 meta
    ap.add_argument("--num-subs", type=int, default=1)
    ap.add_argument("--debug", type=int, nargs="+", default=None, help="k a1 a2 ...: one episode -> <out>/debug_k{k}.mp4")
    ap.add_argument("--chrome", default=None)
    ap.add_argument("--chrome-args", default="")
    args = ap.parse_args()
    root = Path(args.out)
    chrome, chrome_args = args.chrome or find_chrome(), shlex.split(args.chrome_args)

    if args.debug:
        k, swaps = args.debug[0], args.debug[1:]
        ep = episode(list(range(k)), swaps)
        root.mkdir(parents=True, exist_ok=True)
        (root / f"debug_k{k}.json").write_text(json.dumps(ep))
        return render([{"id": 0, "start": ep["start"], "starts": ep["starts"], "swaps": ep["swaps"],
                        "out": str(root / f"debug_k{k}.mp4")}], 1, chrome, chrome_args, PAGE)

    todo = []
    for k in args.ks:
        for name, eps in build(k, args.counts, args.test_ns, args.test_num, args.seed).items():
            d = root / name
            for i, ep in enumerate(eps):
                ep["video"] = f"videos/{i // 1000:02d}/{i:06d}.mp4"
            if args.shard == 0 and args.sub == 0:  # 每个 shard 用同一个 seed 造出同样的 split，只有 shard 0 写 meta
                d.mkdir(parents=True, exist_ok=True)
                with open(d / "meta.jsonl", "w") as f:
                    for i, ep in enumerate(eps):
                        f.write(json.dumps({"id": i, **ep}) + "\n")
            todo += [(d, ep) for ep in eps]
    if args.shard == 0 and args.sub == 0:
        (root / "info.json").write_text(json.dumps({
            "source": "auto_research/datagen/gen_multi.py + render_multi.html (ShellGame renderer with k colored balls)",
            "palette": ["red e63946", "green 2ecc40", "blue 1f6feb", "yellow ffd60a", "magenta b83dd6"],
            "ks": args.ks, "counts_by_num_swaps": args.counts, "test_ns": args.test_ns, "test_num": args.test_num,
            "seed": args.seed,
            "labels": "balls[c][b] = slot of ball b at the end of chunk c; ball/start/final = ball 0 (red), as in ShellGame",
            "latents": "datagen/encode_latents.py, same format as ShellGame"}, indent=2))
    tasks = [{"id": i, "start": ep["start"], "starts": ep["starts"], "swaps": ep["swaps"], "out": str(d / ep["video"])}
             for i, (d, ep) in enumerate(todo[args.shard::args.num_shards][args.sub::args.num_subs])
             if not (d / ep["video"]).exists()]
    print(f"shard {args.shard}/{args.num_shards} sub {args.sub}/{args.num_subs}: {len(tasks)} episodes to render", flush=True)
    render(tasks, args.workers, chrome, chrome_args, PAGE)


if __name__ == "__main__":
    main()
