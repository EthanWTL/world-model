# 把所有 run 的 log.jsonl 汇总成一张表：最新的 eval、训练 loss、fast weight 改了多少、probe 能不能读出球
#   python summary.py              # 所有 run
#   python summary.py dot_ e2e_    # 名字里带这些的
#   python summary.py --curve dot_both_all   # 一个 run 的 eval 曲线
import argparse
import json
from pathlib import Path

ROOT = Path("/project/sds-rise/ethan/Finetune/world/auto_research")


def load(run):
    rows = [json.loads(l) for l in open(ROOT / run / "log.jsonl")]
    train = [r for r in rows if "loss" in r]
    evals = [r for r in rows if any(k.startswith("eval/") for k in r)]
    return train, evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filters", nargs="*")
    ap.add_argument("--curve", default=None)
    args = ap.parse_args()
    if args.curve:
        _, evals = load(args.curve)
        keys = ["n05_tf_acc", "n05_tf_p", "n10_tf_acc", "n10_tf_p", "n20_tf_acc", "n20_tf_p", "n05_tf_mse1"]
        print("step   " + " ".join(f"{k:>11s}" for k in keys))
        for r in evals:
            print(f"{r['step']:6d} " + " ".join(f"{r.get('eval/' + k, float('nan')):11.3f}" for k in keys))
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
        g = lambda k: e.get("eval/" + k, float("nan"))
        print(f"{run:24s} {t.get('step', 0):5d} {avg('loss'):7.4f} {avg('loss_last'):7.4f} {avg('dW_up_mean'):8.1e} "
              f"{avg('dW_down_mean'):8.1e} {probe:>22s} | {e.get('step', 0):5d} {g('curr3_tf_acc'):5.2f} {g('curr3_tf_p'):4.2f} {g('n05_tf_acc'):6.2f} {g('n05_tf_p'):5.2f} "
              f"{g('n10_tf_acc'):6.2f} {g('n10_tf_p'):5.2f} {g('n20_tf_acc'):6.2f} {g('n20_tf_p'):5.2f}")


if __name__ == "__main__":
    main()
