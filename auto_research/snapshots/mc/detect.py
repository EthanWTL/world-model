# Ball detector from arXiv 2608.30692, Appendix D.3: split the reveal frame into five equal-width regions,
# count red pixels with r - max(g, b) > 0.18, and pick the strongest region (-1 if there is no red at all).
# Usage: python shellgame/detect.py [split ...]   checks rendered videos against meta.jsonl
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:  # 多球检测器用 torch（evaluate 里在 GPU 上批量算）；只检查单球视频时没有 torch 也能跑
    import torch
except ImportError:
    torch = None


def ball_slot(frame, num_cups=5):
    """frame: [H, W, 3] RGB, uint8 or float in [0, 1]. Returns the slot index of the ball, or -1."""
    f = np.asarray(frame, dtype=np.float32)
    if f.max() > 1.5:
        f = f / 255.0
    red = (f[..., 0] - np.maximum(f[..., 1], f[..., 2])) > 0.18
    counts = [r.sum() for r in np.array_split(red, num_cups, axis=1)]
    return int(np.argmax(counts)) if max(counts) > 0 else -1


# ---- 多球（Dataset/ShellGameMulti，datagen/render_multi.html）：球 b 的颜色 = PALETTE[b]，按 HSV 色相分 ----
# 渲染帧里的色相中位数：红 356、绿 125、蓝 217、黄 49、紫 288；桌面 hue 31–34、S≈0.40（阴影/边缘到 ~0.55），
# 和黄色只差 ~15°，所以黄色的 S 门槛最高。阈值在渲染帧和 VAE 往返帧上标定过（LOG.md 多球那一节）
BALL_HUE = [(330, 20), (95, 160), (190, 245), (40, 70), (265, 320)]  # 色相窗口 [lo, hi]（度，红色跨 0）
BALL_SMIN = [0.5, 0.45, 0.5, 0.65, 0.45]
BALL_VMIN = [0.3, 0.25, 0.3, 0.5, 0.25]


def hsv(f):
    # f: torch [..., 3] in [0, 1] -> (色相 [0, 360), 饱和度, 亮度)
    mx, mn = f.max(-1).values, f.min(-1).values
    d = (mx - mn).clamp_min(1e-6)
    r, g, b = f.unbind(-1)
    h = torch.where(mx == r, ((g - b) / d) % 6, torch.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, (mx - mn) / mx.clamp_min(1e-6), mx


def ball_weight(f, k):
    # f: torch [..., H, W, 3] in [0, 1] -> [..., k, H, W]：像素属于球 b 的程度（在色相窗口里且够亮时 = S - S_min，否则 0）
    h, s, v = hsv(f)
    out = []
    for b in range(k):
        lo, hi = BALL_HUE[b]
        inside = (h >= lo) & (h <= hi) if lo < hi else (h >= lo) | (h <= hi)
        out.append(torch.where(inside & (v > BALL_VMIN[b]), (s - BALL_SMIN[b]).clamp_min(0), torch.zeros_like(s)))
    return torch.stack(out, -3)


def color_mass(frames, k, num_cups=5):
    # frames: torch [B, T, H, W, 3] in [0, 1] -> [B, k, 5]：每个球的颜色在 5 个位置各有多少（T 帧加起来）
    w = ball_weight(frames, k).sum(1)                                     # [B, k, H, W]
    return torch.stack([c.sum((-1, -2)) for c in w.tensor_split(num_cups, dim=-1)], -1)


def ball_slots(frame, k, num_cups=5, min_px=3):
    """frame: [H, W, 3] RGB（uint8 或 [0, 1] float）。返回 k 个球各自的位置（像素数最多的那一列；不到 min_px 个像素 = -1）"""
    f = frame.float() if torch.is_tensor(frame) else torch.as_tensor(np.asarray(frame, dtype=np.float32))
    if f.max() > 1.5:
        f = f / 255.0
    w = ball_weight(f, k) > 0                                             # [k, H, W]
    counts = torch.stack([c.sum((-1, -2)) for c in w.tensor_split(num_cups, dim=-1)], -1)  # [k, 5]：和 np.array_split 一样分 5 列
    return [int(c.argmax()) if c.max() >= min_px else -1 for c in counts]


def read_frames(path, idx):
    cap = cv2.VideoCapture(str(path))
    frames, i = {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in idx:
            frames[i] = f[..., ::-1]
        i += 1
    return frames, i


if __name__ == "__main__":
    root = Path("/project/sds-rise/ethan/Dataset/ShellGame")
    splits = sys.argv[1:] or ["train"] + [f"test/n{n:02d}" for n in range(5, 31)]
    for split in splits:
        checked = bad = 0
        for line in open(root / split / "meta.jsonl"):
            ep = json.loads(line)
            path = root / split / ep["video"]
            if not path.exists():
                continue
            last = ep["num_frames"] - 1
            frames, n = read_frames(path, {16, last})
            ok = n == ep["num_frames"] and ball_slot(frames[16]) == ep["start"] and ball_slot(frames[last]) == ep["final"]
            checked += 1
            if not ok:
                bad += 1
                print(f"MISMATCH {split} {ep['id']}: frames {n}/{ep['num_frames']}")
        print(f"{split}: {checked} videos checked, {bad} bad")
