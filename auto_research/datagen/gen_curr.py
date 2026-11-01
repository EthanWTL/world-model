# 课程数据：N = 0..4 次交换（主训练集只有 N=5），渲染到 Dataset/ShellGameCurr/train，用 generate.py（拷贝）里的函数。
# 不重复的 episode 数有上限：N 次交换有 5·10^N 种，所以 N=0/1/2 只能有 5/50/500 个，训练时有放回地抽
#   sbatch render_curr.slurm   （CPU，standard 分区）
import argparse
import json
import shlex
from pathlib import Path

import numpy as np

from generate import episode, find_chrome, render, sample

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/project/sds-rise/ethan/Dataset/ShellGameCurr")
ap.add_argument("--counts", type=int, nargs="+", default=[5, 50, 500, 4000, 4000])  # N = 0, 1, 2, 3, 4
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--chrome", default=None)
ap.add_argument("--chrome-args", default="")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
taken = set()
eps = [episode(e[0], e[1:]) for n, c in enumerate(args.counts) for e in sample(rng, n, c, taken)]
d = Path(args.out) / "train"
for i, ep in enumerate(eps):
    ep["video"] = f"videos/{i // 1000:02d}/{i:06d}.mp4"
if args.shard == 0:
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "meta.jsonl", "w") as f:
        for i, ep in enumerate(eps):
            f.write(json.dumps({"id": i, **ep}) + "\n")
    (Path(args.out) / "info.json").write_text(json.dumps({
        "source": "auto_research/datagen/gen_curr.py (copy of world_model/shellgame/generate.py + render.html)",
        "counts_by_num_swaps": args.counts, "seed": args.seed,
        "note": "same renderer/format as Dataset/ShellGame; N+3 chunks per episode"}, indent=2))
tasks = [{"id": k, "start": ep["start"], "swaps": ep["swaps"], "out": str(d / ep["video"])}
         for k, ep in enumerate(eps[args.shard::args.num_shards]) if not (d / ep["video"]).exists()]
print(f"shard {args.shard}/{args.num_shards}: {len(tasks)} episodes to render", flush=True)
render(tasks, args.workers, args.chrome or find_chrome(), shlex.split(args.chrome_args))
