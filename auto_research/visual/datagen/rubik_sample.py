# 魔方小样（2026-11-03 给用户看效果用的）：抽几个 episode -> 渲染 -> VAE 编码 + 解码 -> meta + latent + 预览（mp4 / gif /
# 每个 chunk 末帧的拼图，右边并排 VAE 解码）；顺便看 VAE 往返后每个看得见的格子颜色还认不认得出（cell_acc）
#   python rubik_sample.py --sample 6 --moves 0 2 4 8 12 16 --out <DATA>/RubikMemory/sample
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rubik import ACTIONS, FPS, H, W, RubikScene, episode, render_episode  # noqa: E402


@torch.no_grad()
def cell_acc(label_scene, ep, chunks, frames, min_px=3):
    # frames: [len(chunks), H, W, 3] float [0, 1]（这些 chunk 的末帧）-> 每帧看得见的格子里颜色认对的比例、看得见的格子数
    _, ends = label_scene.cube.transforms(ep["actions"])
    tex = torch.as_tensor(np.asarray(ep["textures"]), device=label_scene.device).long()
    label_scene.grid = G = tex.shape[-1]
    sid, cell, frac, shade, _ = label_scene.trace(ends[chunks])
    inner = (sid >= 0) & ((frac - 0.5).abs() < 0.3).all(-1)                 # 只看格子中间那块（离格线远）
    key = (sid * G + cell[..., 1]) * G + cell[..., 0]
    accs, ns = [], []
    for f in range(len(chunks)):
        m = inner[f]
        k, c, sh = key[f][m], frames[f].reshape(-1, 3)[m], shade[f][m]
        S = tex.numel()
        cnt = torch.zeros(S, device=k.device).index_add_(0, k, torch.ones_like(sh))
        col = torch.zeros(S, 3, device=k.device).index_add_(0, k, c)
        shd = torch.zeros(S, device=k.device).index_add_(0, k, sh)
        ok = cnt >= min_px
        col, shd = col[ok] / cnt[ok, None], shd[ok] / cnt[ok]
        pred = ((col[:, None] - shd[:, None, None] * label_scene.pal[None]) ** 2).sum(-1).argmin(-1)
        accs.append((pred == tex.flatten()[ok]).float().mean().item())
        ns.append(int(ok.sum()))
    return accs, ns


def preview(ep, gt, dec, out, name):
    # mp4 / gif：顶上阶段色带 + 文字；右边并排 VAE 解码；_chunks.png：每个 chunk 的末帧
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=13)
    K, N = ep["num_chunks"], ep["moves"]

    def phase(c):
        if c == 0:
            return "hold", (120, 120, 120)
        if c <= 3:
            return f"display {c}/3", (40, 170, 60)
        if c < K - 3:
            return f"scramble {c - 3}/{N}", (235, 104, 52)
        return f"final display {c - K + 4}/3 (eval)", (240, 200, 30)

    def chunk_of(f):
        return 0 if f <= 16 else 1 + (f - 17) // 20

    frames = []
    for f in range(len(gt)):
        c = chunk_of(f)
        panels = [gt[f]] + ([dec[f]] if dec is not None else [])
        img = Image.fromarray(np.concatenate([np.full((22, W * len(panels), 3), 255, np.uint8),
                                              np.concatenate(panels, 1)], 0))
        d = ImageDraw.Draw(img)
        p, col = phase(c)
        d.rectangle([0, 0, W * len(panels) - 1, 5], fill=col)
        d.text((4, 6), f"chunk {c}/{K - 1}  {p}  action: {ep['action_names'][c]}", fill=(20, 20, 20), font=font)
        for i, s in enumerate(["render (GT)", "VAE decode of stored latent"][:len(panels)]):
            d.text((W * i + 5, 22 + H - 18), s, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        frames.append(np.asarray(img))
    imageio.mimwrite(out / f"{name}.mp4", frames, fps=FPS, quality=9, macro_block_size=1)
    gif = [Image.fromarray(f).quantize(colors=128, dither=Image.Dither.NONE) for f in frames[::2]]
    gif[0].save(out / f"{name}.gif", save_all=True, append_images=gif[1:], duration=1000 * 2 // FPS, loop=0)
    tiles = []
    for c in range(K):
        img = Image.new("RGB", (W, H + 34), (255, 255, 255))
        img.paste(Image.fromarray(gt[16 + 20 * c]), (0, 34))
        d = ImageDraw.Draw(img)
        p, col = phase(c)
        d.rectangle([0, 0, W - 1, 4], fill=col)
        d.text((4, 7), f"chunk {c}  {p}", fill=(20, 20, 20), font=font)
        d.text((4, 20), f"after action {ep['action_names'][c]}", fill=(20, 20, 20), font=font)
        tiles.append(np.asarray(img))
    per = 8
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % per)
    sheet = np.concatenate([np.concatenate(tiles[i:i + per], 1) for i in range(0, len(tiles), per)], 0)
    Image.fromarray(sheet).save(out / f"{name}_chunks.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=6)
    ap.add_argument("--moves", type=int, nargs="+", default=[0, 2, 4, 8, 12, 16])
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decode", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    out = Path(args.out)
    (out / "preview").mkdir(parents=True, exist_ok=True)
    scene = RubikScene(args.n, device="cuda")
    labels = RubikScene(args.n, ssaa=1, device="cuda")
    rng = np.random.default_rng(args.seed)
    eps = [{"id": i, **episode(rng, args.moves[i % len(args.moves)], args.n, args.grid)} for i in range(args.sample)]
    with open(out / "meta.jsonl", "w") as f:
        for ep in eps:
            f.write(json.dumps(ep) + "\n")
    (out / "info.json").write_text(json.dumps({
        "source": "auto_research/visual/datagen/rubik.py (torch ray-cast renderer, 4x4 SSAA)", "size": [W, H], "fps": FPS,
        "n": args.n, "grid": args.grid, "colors": 8, "seed": args.seed, "actions": ACTIONS,
        "episode": "chunk 0 hold, chunks 1-3 display (y y x), N random face turns, final display (y y x); "
                   "evaluate the end frames of the last scramble chunk and the 3 display chunks",
        "latents": "float16 [16, 5K, 24, 32], Wan2.1 VAE mode, (mu - latents_mean) / latents_std"}, indent=2))
    if args.decode:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from vae_io import encode, latent_path, load_vae
        vae, mean, std = load_vae()
    for ep in eps:
        gt = render_episode(scene, ep)
        dec, msg = None, ""
        if args.decode:
            z = encode(vae, mean, std, [gt])
            p = latent_path(out, "", ep["id"])
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, z[0].cpu().numpy())
            x = vae.decode(z.float() * std + mean).sample
            dec = ((x.clamp(-1, 1) + 1) * 127.5).round().byte()[0].permute(1, 2, 3, 0).cpu().numpy()
            ch = ep["eval_chunks"]
            idx = [16 + 20 * c for c in ch]
            a_gt, n_gt = cell_acc(labels, ep, ch, torch.from_numpy(gt[idx]).cuda().float() / 255)
            a_dec, _ = cell_acc(labels, ep, ch, torch.from_numpy(dec[idx]).cuda().float() / 255)
            mse = ((gt[idx].astype(np.float32) - dec[idx].astype(np.float32)) ** 2).mean()
            msg = (f" | eval frames: visible cells {n_gt}, cell_acc GT {np.mean(a_gt):.3f} VAE {np.mean(a_dec):.3f} "
                   f"({' '.join(f'{a:.3f}' for a in a_dec)}), VAE PSNR {10 * math.log10(255 ** 2 / mse):.1f} dB")
        name = f"ep{ep['id']:03d}_moves{ep['moves']:02d}"
        preview(ep, gt, dec, out / "preview", name)
        print(f"{name}: {ep['num_chunks']} chunks, {ep['num_frames']} frames, actions {' '.join(ep['action_names'])}{msg}",
              flush=True)


if __name__ == "__main__":
    main()
