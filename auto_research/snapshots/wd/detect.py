# Ball detector from arXiv 2608.30692, Appendix D.3: split the reveal frame into five equal-width regions,
# count red pixels with r - max(g, b) > 0.18, and pick the strongest region (-1 if there is no red at all).
# Usage: python shellgame/detect.py [split ...]   checks rendered videos against meta.jsonl
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def ball_slot(frame, num_cups=5):
    """frame: [H, W, 3] RGB, uint8 or float in [0, 1]. Returns the slot index of the ball, or -1."""
    f = np.asarray(frame, dtype=np.float32)
    if f.max() > 1.5:
        f = f / 255.0
    red = (f[..., 0] - np.maximum(f[..., 1], f[..., 2])) > 0.18
    counts = [r.sum() for r in np.array_split(red, num_cups, axis=1)]
    return int(np.argmax(counts)) if max(counts) > 0 else -1


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
