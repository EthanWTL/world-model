# 第一轮的本地排队（2026-11-03）：Slurm 排不上（账号 fairshare 0.08，我们 94 个任务在排，最早的已经排了 28 小时），
# 就在这个方向自己的两张本地卡上跑完一个接下一个（用户："本地这个任务可以用两张卡 … 等到finetune的时候只能用两张卡"）
#   每张卡同一时间 1 个训练 run：方向一测过，一张 H200 上并排几个 run 总吞吐不变（auto_research/LOG.md 第 20 条），
#     一个一个跑，排在前面的 run 反而先出完整结果；评估（采样 batch 小，喂不饱卡）另开 1 个槽，跟训练并排跑
#   队列 = r1.sh 写的 r1_queue.tsv，一行一个：名字 \t train|eval \t 完成标志文件 \t 参数。每轮都重读，可以随时改顺序 / 加行 / 删行
#     已经有结果 -> 跳过；Slurm 上在跑 -> 先不管（要是挂了没出结果，之后本地接着跑，train.py 从 last.pt 续）；
#     Slurm 上还在排队 -> 取消 Slurm 的，改在本地跑；同一个本地起了 2 次还没结果 -> 放弃，写进日志
#   2/3 是这个方向一直占着的卡：别人叠上来也不让（用户给方向一 6/7 定的规则），只看空显存够不够一个 run，免得 OOM
#   本地起的进程带环境变量 VQ_ITEM / VQ_KIND，按它数我们在每张卡上有什么（队列自己挂了重起也认得出来）
#   nohup python waves/local_queue.py --gpus 2 3 > /project/sds-rise/ethan/Finetune/world/auto_research/visual/logs/local_queue.log 2>&1 &
import argparse
import os
import subprocess
import time
from pathlib import Path

R = Path("/project/sds-rise/ethan/Finetune/world/auto_research/visual")
CODE = Path(__file__).resolve().parent.parent / "snapshots" / "r1"
PY = "/project/sds-rise/ethan/miniforge3/envs/world/bin/python"
SCRIPT = {"train": "train.py", "eval": "evaluate.py"}
NEED_GB = {"train": 45, "eval": 30}  # 起之前卡上至少空这么多（r1 run B=8 估 ~30 GB，base 评估 ~20 GB）
MAX_TRIES = 2


def now():
    return time.strftime("%F %T")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def ours():
    # 我们本地起的进程：[(VQ_ITEM, VQ_KIND, GPU)]（dataloader 的子进程也带这些变量，所以下面都用集合）
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if not (p / "comm").read_text().startswith("python"):
                continue
            env = dict(kv.split(b"=", 1) for kv in (p / "environ").read_bytes().split(b"\0") if b"=" in kv)
        except OSError:  # 别人的进程读不了 / 进程刚结束
            continue
        if b"VQ_ITEM" in env:
            out.append((env[b"VQ_ITEM"].decode(), env[b"VQ_KIND"].decode(), env.get(b"CUDA_VISIBLE_DEVICES", b"").decode()))
    return out


def free_gb(g):
    return int(sh(f"nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i {g}").strip() or 0) / 1024


def slurm(name):
    return [tuple(l.split()) for l in sh(f"squeue -u jqf8qm -h -n {name} -o '%i %T'").splitlines() if l.strip()]


def read_queue(path):
    items = []
    for l in Path(path).read_text().splitlines():
        if l.strip() and not l.startswith("#"):
            name, kind, done, args = (l.split("\t") + [""] * 3)[:4]
            items.append((name, kind, Path(done), args))
    return items


def launch(g, name, kind, args):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g), "VQ_ITEM": name, "VQ_KIND": kind, "PYTHONNOUSERSITE": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"}
    log = open(R / "logs" / f"{name}_local.log", "a")
    return subprocess.Popen(f"exec {PY} -u {SCRIPT[kind]} {args} --gpu 0", shell=True, cwd=CODE, env=env, stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--queue", default=str(R / "r1_queue.tsv"))
    ap.add_argument("--every", type=int, default=60)
    args = ap.parse_args()
    tries, procs, gave_up = {}, [], set()
    print(f"{now()} local queue on GPUs {args.gpus}: {args.queue}", flush=True)
    while True:
        procs = [p for p in procs if p.poll() is None]  # 收掉跑完的子进程
        run = ours()
        busy = {(k, g) for _, k, g in run}  # (类型, 卡) 这个槽已经有我们的进程
        here = {n for n, _, _ in run}
        todo = [it for it in read_queue(args.queue) if not it[2].exists()]
        for name, *_ in todo:
            if tries.get(name, 0) >= MAX_TRIES and name not in here and name not in gave_up:
                gave_up.add(name)
                print(f"{now()} {name}: started {MAX_TRIES} times locally, still no result -> give up (see logs/{name}_local.log)",
                      flush=True)
        if all(n in gave_up for n, *_ in todo):
            print(f"{now()} queue finished" + (f", gave up: {sorted(gave_up)}" if gave_up else ""), flush=True)
            return
        for g in args.gpus:
            for kind in ("train", "eval"):
                if (kind, str(g)) in busy:
                    continue
                for name, k, done, a in todo:
                    if k != kind or name in here or name in gave_up:
                        continue
                    st = slurm(name)
                    if any(s in ("RUNNING", "COMPLETING", "CONFIGURING") for _, s in st):
                        continue  # Slurm 上在跑
                    fg = free_gb(g)
                    if fg < NEED_GB[kind]:
                        print(f"{now()} GPU {g}: {fg:.0f} GB free < {NEED_GB[kind]} GB, {name} waits", flush=True)
                        break
                    if st:
                        sh(f"scancel {' '.join(j for j, _ in st)}")
                        print(f"{now()} {name}: cancelled pending Slurm {[j for j, _ in st]}", flush=True)
                    procs.append(launch(g, name, kind, a))
                    tries[name] = tries.get(name, 0) + 1
                    here.add(name)
                    busy.add((kind, str(g)))
                    print(f"{now()} GPU {g} ({fg:.0f} GB free): {kind} {name} (try {tries[name]})", flush=True)
                    time.sleep(90)  # 等它分到显存，这张卡的下一个槽再看空显存
                    break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
