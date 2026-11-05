# wave MC 的本地排队（用户：这个任务最多两张本地卡 = GPU 6 / 7）：
# 每张卡最多 --slots 个我们的 run；卡上还有 ≥ --free-gb 空显存才起下一个格子（一个 run 峰值 ~20 GB，不够就等，免得 OOM）。
# 别人叠上来的不让（用户："这67如果是别人叠上来的 就不要给他们让开 我们也是正常使用"）。
# 顺序：开始过的格子（有 last.pt）先续完，其余按 k1 s0, k5 s0, k2 s0, k3 s0, k1 s1, k5 s1, k2 s1, k3 s1。
# 每个格子：最终评估的 json 已经有了 -> 跳过；本地或 Slurm 上正在跑 -> 跳过；
#   否则取消 Slurm 上排队的训练和评估，用 mc.sh 在本地起。train.py 有 last.pt 就从那里续（每 500 步存一次），
#   已经到 20000 步就直接退出、接着跑评估。
# --stop：先停排队脚本（不然 5 分钟内它会把空出来的槽补上），再停本地卡上这个任务的训练 / 评估（--gpus 只停这几张）；
#   正在写 .pt 的等它写完再停（停在写的中间 last.pt 就坏了、续不上）；每个 run 停在哪一步写进 logs/local_queue.log。
# 2026-11-03 起这个任务不用本地卡（用户要 6/7 跑 paper："这样我们这个任务就完全不去用本地gpu了"），只走 Slurm。
#   以后要用本地卡，等用户说了、看过 nvidia-smi 再起：
#   nohup python waves/local_queue.py --gpus 6 7 >> /project/sds-rise/ethan/Finetune/world/auto_research/logs/local_queue.log 2>&1 &
import argparse
import os
import pickle
import re
import signal
import subprocess
import time
import zipfile
from pathlib import Path

ORDER = [(1, 0), (5, 0), (2, 0), (3, 0), (1, 1), (5, 1), (2, 1), (3, 1)]  # (k, seed)
RS = (16, 72, 288, 1152)
R = Path("/project/sds-rise/ethan/Finetune/world/auto_research")
LOG = R / "logs" / "local_queue.log"
HERE = Path(__file__).resolve().parent.parent


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def name(k, r, s):
    return f"mbc_k{k}_r{r}_s{s}"


def comm(pid):
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return ""


def procs():
    # 本地正在跑的 wave MC 进程 [(pid, 格子名, GPU)]：只算 python 进程（不算 sh -c 外壳），
    # 参数里有格子名（train.py 的 --run，evaluate.py 的路径）
    out = []
    for pid in sh("pgrep -f 'mbc_k'").split():
        if not comm(pid).startswith("python"):
            continue
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
            env = Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        gpu = next((e.split("=", 1)[1] for e in env if e.startswith("CUDA_VISIBLE_DEVICES=")), None)
        n = next((p for a in argv for p in a.split("/") if p.startswith("mbc_k")), None)
        if n:
            out.append((int(pid), n, gpu))
    return out


def ours():
    return {n: g for _, n, g in procs()}


def free_gb(g):
    return int(sh(f"nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i {g}").strip() or 0) / 1024


def slurm(n):
    return [tuple(l.split()) for l in sh(f"squeue -u jqf8qm -h -n {n} -o '%i %T'").splitlines() if l.strip()]


def done(n):
    return (R / n / "eval" / "ckpt_020000_ema.json").exists()


def take(g, k, r, s):
    # 起一个格子；返回 False = 这个格子不用本地跑（已经评完 / 本地或 Slurm 上正在跑）
    n = name(k, r, s)
    jobs = slurm(n) + slurm(f"ev_{n}")
    if done(n) or n in ours() or any(st == "RUNNING" for _, st in jobs):
        print(f"  {n}: done / running, skip", flush=True)
        return False
    if jobs:
        sh(f"scancel {' '.join(j for j, _ in jobs)}")
        print(f"  {n}: cancelled Slurm {[j for j, _ in jobs]}", flush=True)
    print(sh(f"cd {HERE} && MODE=local GPU={g} KS={k} RS={r} SEEDS={s} bash waves/mc.sh").strip(), flush=True)
    return True


def say(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def saving(pid):
    # 这个进程开着 .pt 文件吗（torch.save 正在写 last.pt / ckpt_*.pt）
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return False
    for fd in fds:
        try:
            if os.readlink(f"/proc/{pid}/fd/{fd}").endswith(".pt"):
                return True
        except OSError:
            pass
    return False


class _Stub:  # 读 .pt 的元数据时代替 torch 的类（张量全跳过）
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


def ckpt_step(path):
    # 不用 torch 读出 .pt 里存的步数：只解 zip 里的 data.pkl；文件没写完（坏了）返回 None
    class Skip(pickle.Unpickler):
        def find_class(self, mod, cls):
            return _Stub if mod.split(".")[0] == "torch" else super().find_class(mod, cls)

        def persistent_load(self, pid):
            return None

    try:
        with zipfile.ZipFile(path) as z:
            pkl = next(f for f in z.namelist() if f.endswith("data.pkl"))
            return Skip(z.open(pkl)).load().get("step")
    except Exception:
        return None


def logged_step(n):
    # 训练日志里最后打印的步数（停的时候跑到了哪）
    try:
        txt = (R / "logs" / f"{n}_local.log").read_text(errors="replace").replace("\r", "\n")
    except OSError:
        return None
    steps = re.findall(r"^step (\d+) \|", txt, re.M)
    return int(steps[-1]) if steps else None


def stop(gpus):
    me = os.getpid()
    for pid in sh("pgrep -f 'local_queue.py'").split():
        if int(pid) != me and comm(pid).startswith("python"):
            os.kill(int(pid), signal.SIGTERM)
            say(f"stop: feeder {pid} stopped")
    runs = {pid: n for pid, n, g in procs() if gpus is None or g in {str(x) for x in gpus}}
    todo, t0 = dict(runs), time.time()
    while todo:  # 正在写 .pt 的等它写完（最多等 10 分钟）
        for pid in list(todo):
            if not saving(pid) or time.time() - t0 > 600:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                del todo[pid]
        time.sleep(1)
    for _ in range(120):
        if not any(Path(f"/proc/{pid}").exists() for pid in runs):
            break
        time.sleep(1)
    for pid, n in sorted(runs.items(), key=lambda x: x[1]):
        alive = ", STILL ALIVE" if Path(f"/proc/{pid}").exists() else ""
        say(f"stop: {n} pid {pid}{alive} | logged step {logged_step(n)} | last.pt step {ckpt_step(R / n / 'last.pt')}")
    where = "all GPUs" if gpus is None else f"GPUs {gpus}"
    say(f"stop: {len(runs)} wave MC processes stopped on {where}" if runs else f"stop: nothing running on {where}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=None, help="排队用哪几张卡（默认 6 7）；--stop 时只停这几张（默认全部）")
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--free-gb", type=float, default=24)
    ap.add_argument("--dry", action="store_true", help="只打印顺序和本地正在跑的，不起任何东西")
    ap.add_argument("--stop", action="store_true", help="停掉这个任务在本地卡上的一切：排队脚本、训练、评估")
    args = ap.parse_args()
    if args.stop:
        return stop(args.gpus)
    gpus = args.gpus or [6, 7]
    cells = [(k, r, s) for k, s in ORDER for r in RS]
    queue = sorted(cells, key=lambda c: not (R / name(*c) / "last.pt").exists())  # 稳定排序：开始过的在前，其余顺序不变
    queue = [c for c in queue if not done(name(*c))]
    print(f"{time.strftime('%F %T')} {len(queue)} cells left on GPUs {gpus}, {args.slots} slots each:\n  "
          + " ".join(name(*c) for c in queue) + f"\n  running locally: {ours()}", flush=True)
    if args.dry:
        return
    while queue:
        for g in gpus:
            while queue and sum(v == str(g) for v in ours().values()) < args.slots and free_gb(g) >= args.free_gb:
                k, r, s = queue.pop(0)
                print(f"{time.strftime('%F %T')} GPU {g}: {free_gb(g):.0f} GB free -> {name(k, r, s)}", flush=True)
                if take(g, k, r, s):
                    time.sleep(90)  # 等它分到显存，下一个格子再看空显存
        time.sleep(300)
    print(f"{time.strftime('%F %T')} queue finished", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()
