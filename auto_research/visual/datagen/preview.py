# 看数据长什么样：数据集只存 latent + meta.jsonl（视频不落盘，评测时按 meta 精确重渲染），这里把存好的 episode 渲染出来，加阶段标注
#   每个 episode 三个文件：
#     <split>_ep<id>.mp4 / .gif：顶上一条阶段色带 + 文字（绿 = 观察，灰 = 离开 / 间隔（目标面看不见），黄 = 返回）；
#                                --decode 1 时右边并排 = 存的 latent 经 VAE 解码（模型实际学的就是这个）
#     <split>_ep<id>_chunks.png：每个 chunk 的最后一帧（静止、某个面正对）按顺序排开，标 chunk 号 / 阶段 / 正对的面，
#                                目标面正对的帧（观察最后一个 chunk、返回 chunk）加黄框
#   python datagen/preview.py --split test/g04 --num 2 --decode 1 --out <DATA>/CubeMemory/preview
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import ROOT, latent_path  # noqa: E402
from cube import FPS, H, W, Scene, render_episode  # noqa: E402

COL = {"obs": (40, 170, 60), "leave": (120, 120, 120), "gap": (120, 120, 120), "return": (240, 200, 30)}
BAR = 22
FONT = ImageFont.load_default(size=13)


def phase(ep, c):
    return "obs" if c <= 3 else "leave" if c == 4 else "return" if c == ep["num_chunks"] - 1 else "gap"


def text(ep, c):
    p, face, tgt = phase(ep, c), ep["pose"][c][0], ep["target"]
    front = f"front F{face}" + (" = target" if face == tgt else "")
    return {"obs": f"observe {'hold' if c == 0 else f'turn {c}/3'}", "leave": "leave (target turns away)",
            "gap": f"gap {c - 4}/{ep['gap']} (target out of view)", "return": "return (eval: last frame)"}[p] + f" | {front}"


def chunk_of(f):
    return 0 if f <= 16 else 1 + (f - 17) // 20


def compose(ep, gt, dec):
    # 每帧：[阶段色带; GT 渲染 | VAE 解码]
    out = []
    for f in range(len(gt)):
        c = chunk_of(f)
        panels = [gt[f]] + ([dec[f]] if dec is not None else [])
        img = Image.fromarray(np.concatenate([np.full((BAR, W * len(panels), 3), 255, np.uint8),
                                              np.concatenate(panels, 1)], 0))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W * len(panels) - 1, 5], fill=COL[phase(ep, c)])
        d.text((4, 6), f"chunk {c}/{ep['num_chunks'] - 1}  {text(ep, c)}", fill=(20, 20, 20), font=FONT)
        for i, name in enumerate(["render (GT)", "VAE decode of stored latent"][:len(panels)]):
            d.text((W * i + 5, BAR + H - 18), name, fill=(255, 255, 255), font=FONT, stroke_width=2, stroke_fill=(0, 0, 0))
        out.append(np.asarray(img))
    return out


def chunk_sheet(ep, gt, per_row=8):
    K = ep["num_chunks"]
    tiles = []
    for c in range(K):
        img = Image.new("RGB", (W + 8, H + 8 + 34), (255, 255, 255))
        hit = ep["pose"][c][0] == ep["target"]
        if hit:
            img.paste((240, 200, 30), (0, 34, W + 8, H + 8 + 34))
        img.paste(Image.fromarray(gt[16 + 20 * c]), (4, 38))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W + 7, 4], fill=COL[phase(ep, c)])
        d.text((4, 8), f"chunk {c}  {text(ep, c).split(' | ')[0]}", fill=(20, 20, 20), font=FONT)
        d.text((4, 21), f"front F{ep['pose'][c][0]}" + ("  <- TARGET" if hit else ""), fill=(20, 20, 20), font=FONT)
        tiles.append(np.asarray(img))
    rows = []
    for i in range(0, K, per_row):
        row = tiles[i:i + per_row]
        row += [np.full_like(tiles[0], 255)] * (per_row - len(row)) if K > per_row else []
        rows.append(np.concatenate(row, 1))
    return np.concatenate(rows, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["test/g04"])
    ap.add_argument("--num", type=int, default=2)
    ap.add_argument("--decode", type=int, default=1)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=f"{ROOT}/preview")
    args = ap.parse_args()
    import imageio.v2 as imageio
    torch.set_grad_enabled(False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = Scene(1, device="cuda")
    vae = None
    if args.decode:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from evaluate import VAE
        vae = VAE()
    for split in args.split:
        eps = [json.loads(l) for l in open(Path(args.root) / split / "meta.jsonl")][:args.num]
        for ep in eps:
            name = f"{split.replace('/', '_')}_ep{ep['id']:03d}"
            gt = render_episode(scene, ep)
            dec = None
            if vae is not None:
                z = torch.from_numpy(np.load(latent_path(args.root, split, ep["id"]))).float().cuda()[None]
                dec = (vae.decode(z)[0].cpu().numpy() * 255).round().astype(np.uint8)
            fr = compose(ep, gt, dec)
            imageio.mimwrite(out / f"{name}.mp4", fr, fps=FPS, quality=9, macro_block_size=1)
            gif = [Image.fromarray(f).quantize(colors=128, dither=Image.Dither.NONE) for f in fr[::2]]
            gif[0].save(out / f"{name}.gif", save_all=True, append_images=gif[1:], duration=1000 * 2 // FPS, loop=0)
            Image.fromarray(chunk_sheet(ep, gt)).save(out / f"{name}_chunks.png")
            print(f"{name}: {ep['num_chunks']} chunks, {len(gt)} frames, gap {ep['gap']} chunks = {ep['gap_frames']} frames "
                  f"target out of view, target F{ep['target']}, actions {ep['actions']}, "
                  f"front per chunk {[p[0] for p in ep['pose']]}", flush=True)


if __name__ == "__main__":
    main()
