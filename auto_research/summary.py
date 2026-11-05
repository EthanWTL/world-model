# 把所有 run 的 log.jsonl 汇总成一张表：最新的 eval、训练 loss、fast weight 改了多少、probe 能不能读出球
#   python summary.py              # 所有 run
#   python summary.py dot_ e2e_    # 名字里带这些的
#   python summary.py --curve dot_both_all   # 一个 run 的 eval 曲线
#   python summary.py --grid mb_q --step 24000   # 多球：k × N 的表（完整评估的 json，seed 平均），对照"没有容量上限"的 acc₁^k
#   python summary.py --cap mbc                  # wave MC：k × r 的学会时间、各预算下最少要几行、最新快评、20000 步 EMA 完整评估
import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path("/project/sds-rise/ethan/Finetune/world/auto_research")


def prefix(rows):
    # 多球 run 的 eval key 带 k 前缀（eval/k3_n05_tf_acc）：找出来，下面统一按单球的名字取
    return next((m.group(1) for r in rows for k in r for m in [re.match(r"eval/(k\d_)", k)] if m), "")


def load(run):
    rows = [json.loads(l) for l in open(ROOT / run / "log.jsonl")]
    train = [r for r in rows if "loss" in r]
    evals = [r for r in rows if any(k.startswith("eval/") for k in r)]
    return train, evals


def grid(name, step, ks=(1, 2, 3, 4, 5), ns=(5, 10, 20, 30)):
    # <name>_k{k}_s{seed}/eval/ckpt_{step}_ema.json；acc = 所有球都对，ball = 每个球单独算
    res = {}
    for k in ks:
        for f in sorted(ROOT.glob(f"{name}_k{k}_s*/eval/ckpt_{step:06d}_ema.json")):
            r = json.load(open(f))["res"]
            for mode in ("free", "tf"):
                for m in ("acc", "ball"):
                    for n in ns:
                        res.setdefault((mode, m, k, n), []).append(r.get(f"k{k}_n{n:02d}_{mode}_{m}", float("nan")))
    for mode in ("free", "tf"):
        for m in ("acc", "ball"):
            print(f"\n{mode} {m} (seed mean, [seeds])" + ("   null = k=1 acc^k" if m == "acc" else ""))
            print("k    " + "".join(f"{'N' + str(n):>22s}" for n in ns))
            for k in ks:
                cells = []
                for n in ns:
                    v = res.get((mode, m, k, n), [])
                    one = res.get((mode, "acc", 1, n), [])
                    null = f" ({(sum(one) / len(one)) ** k:.2f})" if m == "acc" and one and k > 1 else ""
                    cells.append(f"{sum(v) / len(v):.2f} [{' '.join(f'{x:.2f}' for x in v)}]{null}" if v else "-")
                print(f"k{k}   " + "".join(f"{c:>22s}" for c in cells))


def cap(name, thr, ks=(1, 2, 3, 5), rs=(16, 72, 288, 1152), budgets=(4000, 8000, 12000, 16000, 20000)):
    # 学会的时间 = 快评（原始权重、32 个）里 n05 的 tf 全对第一次 ≥ thr 的步数；没到就记成 >最后一步
    cells = {}
    for k in ks:
        for r in rs:
            for d in sorted(ROOT.glob(f"{name}_k{k}_r{r}_s*")):
                if not (d / "log.jsonl").exists():
                    continue
                _, evals = load(d.name)
                key = f"eval/k{k}_n05_tf_acc"
                ev = [e for e in evals if key in e]
                if not ev:
                    continue
                t = next((e["step"] for e in ev if e[key] >= thr), None)
                fin = ROOT / d.name / "eval" / "ckpt_020000_ema.json"
                cells.setdefault((k, r), []).append({"seed": d.name.rsplit("_s", 1)[1], "t": t, "last": ev[-1],
                                                     "fin": json.load(open(fin))["res"] if fin.exists() else None})

    def table(title, fmt):
        print(f"\n{title}")
        print("k (states)  " + "".join(f"{'r' + str(r):>20s}" for r in rs))
        for k in ks:
            print(f"k{k} ({math.perm(5, k):>3d})    "
                  + "".join(f"{fmt(k, cells.get((k, r), [])):>20s}" for r in rs))

    table(f"学会的时间（n05 tf 全对 ≥ {thr} 的第一步；各 seed）",
          lambda k, c: " ".join(str(x["t"]) if x["t"] else f">{x['last']['step']}" for x in c) or "-")
    table("最新快评 n05 全对 / 每个球 / p（各 seed，原始权重）",
          lambda k, c: " ".join(f"{x['last'][f'eval/k{k}_n05_tf_acc']:.2f}/{x['last'].get(f'eval/k{k}_n05_tf_ball', float('nan')):.2f}"
                                for x in c) or "-")
    table("最新快评 n10 全对（各 seed）",
          lambda k, c: " ".join(f"{x['last'].get(f'eval/k{k}_n10_tf_acc', float('nan')):.2f}" for x in c) or "-")
    print(f"\n各预算下最少要几行（任一 seed / 所有 seed 学会）")
    print("k    " + "".join(f"{str(b):>14s}" for b in budgets))
    for k in ks:
        row = []
        for b in budgets:
            ok_any = [r for r in rs if any(x["t"] and x["t"] <= b for x in cells.get((k, r), []))]
            ok_all = [r for r in rs if cells.get((k, r)) and all(x["t"] and x["t"] <= b for x in cells[(k, r)])]
            row.append(f"{min(ok_any) if ok_any else '-'}/{min(ok_all) if ok_all else '-'}")
        print(f"k{k}   " + "".join(f"{c:>14s}" for c in row))
    for mode in ("free", "tf"):
        table(f"20000 步 EMA 完整评估 {mode}：全对 N5/10/20/30（seed 平均）",
              lambda k, c: "/".join(f"{sum(x['fin'][f'k{k}_n{n:02d}_{mode}_acc'] for x in c if x['fin']) / max(1, sum(1 for x in c if x['fin'])):.2f}"
                                    for n in (5, 10, 20, 30)) if any(x["fin"] for x in c) else "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filters", nargs="*")
    ap.add_argument("--curve", default=None)
    ap.add_argument("--grid", default=None)
    ap.add_argument("--step", type=int, default=24000)
    ap.add_argument("--cap", default=None)
    ap.add_argument("--thr", type=float, default=0.9)
    args = ap.parse_args()
    if args.cap:
        return cap(args.cap, args.thr)
    if args.grid:
        return grid(args.grid, args.step)
    if args.curve:
        _, evals = load(args.curve)
        pre = prefix(evals)
        keys = ["curr3_tf_acc", "n05_tf_acc", "n05_tf_p", "n10_tf_acc", "n10_tf_p", "n20_tf_acc", "n20_tf_p", "n05_tf_mse1"]
        print("step   " + " ".join(f"{k:>11s}" for k in keys))
        for r in evals:
            print(f"{r['step']:6d} " + " ".join(f"{r.get('eval/' + pre + k, float('nan')):11.3f}" for k in keys))
        return
    runs = sorted(p.parent.name for p in ROOT.glob("*/log.jsonl"))
    runs = [r for r in runs if not args.filters or any(f in r for f in args.filters)]
    print(f"{'run':24s} {'step':>5s} {'loss':>7s} {'last':>7s} {'dWup':>8s} {'dWdn':>8s} {'probe(c2..c6)':>22s} | "
          f"{'eval@':>5s} {'c3acc':>5s} {'c3p':>4s} {'n05acc':>6s} {'n05p':>5s} {'n10acc':>6s} {'n10p':>5s} {'n20acc':>6s} {'n20p':>5s}")
    for run in runs:
        train, evals = load(run)
        t = train[-1] if train else {}
        # 训练指标取最后 5 条平均（单条很吵）
        last5 = train[-5:]
        avg = lambda k: sum(r.get(k, 0) for r in last5) / max(len(last5), 1)
        probe = (" ".join(f"{avg(f'probe_acc_c{c}'):.2f}" for c in range(2, 7)) if "probe_acc_c2" in t else
                 " ".join(f"{avg(f'probe_acc_n{n}'):.2f}" for n in range(1, 6)) if "probe_acc_n5" in t else "-")
        e = evals[-1] if evals else {}
        pre = prefix(evals)
        g = lambda k: e.get("eval/" + pre + k, float("nan"))
        print(f"{run:24s} {t.get('step', 0):5d} {avg('loss'):7.4f} {avg('loss_last'):7.4f} {avg('dW_up_mean'):8.1e} "
              f"{avg('dW_down_mean'):8.1e} {probe:>22s} | {e.get('step', 0):5d} {g('curr3_tf_acc'):5.2f} {g('curr3_tf_p'):4.2f} {g('n05_tf_acc'):6.2f} {g('n05_tf_p'):5.2f} "
              f"{g('n10_tf_acc'):6.2f} {g('n10_tf_p'):5.2f} {g('n20_tf_acc'):6.2f} {g('n20_tf_p'):5.2f}")


if __name__ == "__main__":
    main()
