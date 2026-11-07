# 汇总：
#   python summary.py                     完整评估（<run>/eval/*.json）的表：每个 run × 每个 gap 的 cell_acc（free / tf）、lpips、姿态错误
#   python summary.py r1_v r1_q           只看名字里带这些的
#   python summary.py --curve r1_v        一个 run 的快评曲线（tf cell_acc vs 步数 / token / GPU 小时）
#   python summary.py --plot r1           回访误差随 gap 的曲线（按 PANELS 分组的小图；ctrl 和 1/8 瞎猜是灰色参照线）
#   python summary.py --task rubik [rk_]  魔方：每个 run × 每个打乱步数 N 的 cell_acc（free / tf，展示 3 帧 / 最后一帧 / 背面 / 正面）
import argparse
import json
from pathlib import Path

from paths import OUT_DIR

ROOT = Path(OUT_DIR)
GAPS = (0, 2, 4, 6, 8, 12, 16, 24, 32)
MOVES = (0, 2, 4, 8, 12, 16, 24)
PANELS = {"r1": {  # 每个小图 ≤ 4 条（固定颜色顺序），ctrl 在每个图里都是灰色参照
    "projection": ["r1_v", "r1_q", "r1_k", "r1_ffn"],
    "combinations": ["r1_qk", "r1_qv", "r1_kv", "r1_qkv"],
    "write rule": ["r1_v", "r1_v_norm", "r1_v_dot", "r1_qkv_norm"],
    "slots (v, fast heads)": ["r1_v_f1", "r1_v", "r1_v_f8", "r1_v_f16"],
    "blocks (v)": ["r1_v_b1", "r1_v", "r1_v_b16"],
}}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # dataviz 默认调色板前 4 个（固定顺序）
GRAY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"


def evals(filters=()):
    # {(run, json 名): res}，最新的 step 在后面
    out = {}
    for f in sorted(ROOT.glob("*/eval/*.json")):
        run = f.parent.parent.name
        if filters and not any(s in run for s in filters):
            continue
        out[(run, f.stem)] = json.load(open(f))["res"]
    return out


def table(filters):
    ev = evals(filters)
    for metric in ("free_cell_acc", "tf_cell_acc", "tf1_cell_acc", "free_lpips", "free_wrong_face", "free_yaw_err"):
        print(f"\n{metric}  (gap in chunks; vae = GT latent decoded)")
        print(f"{'run':38s}" + "".join(f"{'g' + str(g):>7s}" for g in GAPS))
        vae = None
        for (run, stem), r in ev.items():
            vals = [r.get(f"g{g:02d}_{metric}") for g in GAPS]
            if all(v is None for v in vals):
                continue
            vae = vae or [r.get(f"g{g:02d}_vae_{metric.split('_', 1)[1]}") for g in GAPS]
            print(f"{(run + '/' + stem)[:38]:38s}" + "".join(f"{v:7.3f}" if v is not None else f"{'-':>7s}" for v in vals))
        if vae and any(v is not None for v in vae):
            print(f"{'(vae ceiling)':38s}" + "".join(f"{v:7.3f}" if v is not None else f"{'-':>7s}" for v in vae))


def rubik_table(filters, out=None):
    # 魔方：每个 run 的完整评估（eval/*.json）；行 = run，列 = 打乱步数 N；(vae) 行 = GT latent 解码的上限
    ev = evals(filters)
    lines = []
    for metric in ("free_disp_acc", "free_last_acc", "free_disp_hid", "free_disp_vis", "free_scr_acc", "tf_disp_acc",
                   "tf_last_acc", "free_disp_psnr"):
        lines.append(f"\n{metric}  (N = scramble moves; chance 0.125 for acc)")
        lines.append(f"{'run':40s}" + "".join(f"{'N' + str(n):>7s}" for n in MOVES))
        vae = None
        for (run, stem), r in ev.items():
            vals = [r.get(f"n{n:02d}_{metric}") for n in MOVES]
            if all(v is None for v in vals):
                continue
            if not vae or all(v is None for v in vae):  # VAE 上限只有一个 json 里算了（handback eval 只算一次）
                vae = [r.get(f"n{n:02d}_vae_{metric.split('_', 1)[1]}") for n in MOVES]
            lines.append(f"{(run + '/' + stem)[:40]:40s}" + "".join(f"{v:7.3f}" if v is not None else f"{'-':>7s}" for v in vals))
        if vae and any(v is not None for v in vae):
            lines.append(f"{'(vae ceiling)':40s}" + "".join(f"{v:7.3f}" if v is not None else f"{'-':>7s}" for v in vae))
    text = "\n".join(lines)
    print(text)
    if out:
        Path(out).write_text(text + "\n")


def curve(run):
    rows = [json.loads(l) for l in open(ROOT / run / "log.jsonl")]
    ev = [r for r in rows if any(k.startswith("eval/") for k in r)]
    keys = sorted({k for r in ev for k in r if k.endswith(("_tf_cell_acc", "_tf1_cell_acc", "_tf_disp_acc", "_tf_last_acc",
                                                             "_free_disp_acc", "_free_last_acc"))})
    print(f"{'step':>7s} {'Mtok':>8s} {'GPUh':>6s} " + " ".join(f"{k[5:]:>16s}" for k in keys))
    for r in ev:
        print(f"{r['step']:7d} {r.get('tokens', 0) / 1e6:8.1f} {r.get('gpu_hours', 0):6.1f} "
              + " ".join(f"{r.get(k, float('nan')):16.3f}" for k in keys))


def plot(wave, out, metric="free_cell_acc"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = evals((wave,))
    last = {}
    for (run, stem), r in ev.items():  # 每个 run 取最新的 EMA 完整评估
        if stem.endswith("_ema"):
            last[run] = r
    panels = PANELS[wave]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (title, runs) in zip(axes, panels.items()):
        ax.axhline(1 / 8, color=GRAY, lw=1, ls=":")
        ctrl = last.get(f"{wave}_ctrl")
        if ctrl:
            ax.plot(GAPS, [ctrl.get(f"g{g:02d}_{metric}", float("nan")) for g in GAPS], color=GRAY, lw=2, ls="--",
                    label="ctrl (no memory)")
        for c, run in zip(SERIES, runs):
            if run not in last:
                continue
            y = [last[run].get(f"g{g:02d}_{metric}", float("nan")) for g in GAPS]
            name = run[len(wave) + 1:]
            ax.plot(GAPS, y, color=c, lw=2, marker="o", ms=4, label=name)
            ax.annotate(name, (GAPS[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("gap (chunks out of view; frames = 20·gap + 5)", fontsize=8, color=MUTED)
        ax.grid(color="#e6e5e1", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        ax.axvspan(-1, 6.5, color="#f0efec", zorder=0)  # 训练 gap 范围
    axes[0].set_ylabel(f"{metric}  (chance 1/8 dotted)", fontsize=9, color=MUTED)
    axes[0].set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filters", nargs="*")
    ap.add_argument("--curve", default=None)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--metric", default="free_cell_acc")
    ap.add_argument("--out", default=None)
    ap.add_argument("--task", default="cube", choices=["cube", "rubik"])
    args = ap.parse_args()
    if args.task == "rubik" and not args.curve:
        return rubik_table(args.filters, args.out)
    if args.curve:
        return curve(args.curve)
    if args.plot:
        return plot(args.plot, args.out or str(ROOT / f"{args.plot}_{args.metric}.png"), args.metric)
    table(args.filters)


if __name__ == "__main__":
    main()
