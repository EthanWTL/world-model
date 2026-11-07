# 魔方的评测（2026-11-03）：评测帧 = 打乱最后一个 chunk 的末帧 + 最后展示 3 个 chunk 的末帧（都是静止的）
#   展示 y y x 依次露出：(上, 旧右, 旧后) -> (上, 旧后, 旧左) -> (旧后, 旧下, 旧左)：最后一帧全是打乱时朝背面（看不见）的贴纸
#   每个评测帧：看得见的每一格（格子中间那块的平均颜色）认成最近的调色板颜色（按这个面的明暗），和真值比 -> acc；
#     再按"这个贴纸在打乱结束时朝不朝相机"拆开：hid = 朝下 / 后 / 左（打乱时看不见，只能靠记忆），vis = 朝上 / 前 / 右
#   协议：
#     free：给 chunk 0–3（不动 + 开头展示，所有贴纸都看过一遍），之后全部自己生成（主协议）
#     tf  ：除了最后展示的 3 个 chunk 都给 GT（打乱过程看着），只生成最后展示 -> 只考记忆，没有生成误差累积
#     vae ：GT latent 解码（上限）
#   结果 key：n{N}_{模式}_{部分}_{指标}；部分：scr = 打乱最后一帧（tf 里它是 GT，不算），disp = 展示 3 帧平均，last = 展示最后一帧；
#     指标：acc / hid / vis（cell_acc，瞎猜 = 1/8）、psnr（和精确渲染比）
#   python evaluate.py --ckpt .../visual/rubik_base/ema_020000.pt --max-ctx -1     （ckpt 里 task = rubik 时自动用这个）
#   python rubik_eval.py --saved <infer.py 的输出> --out <json>     （评远端推理存下来的 latent：同一套指标，噪声种子也一样）
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parent / "datagen"))
from rubik import RubikScene, render_episode  # noqa: E402
from data import TASKS, CubeDataset  # noqa: E402
from evaluate import VAE  # noqa: E402
from sampling import rollout  # noqa: E402


@torch.no_grad()
def cell_scores(labels, ep, chunks, frames, min_px=3):
    # frames: [len(chunks), H, W, 3] float [0, 1]（这些 chunk 的末帧）-> [len(chunks), 3]：全部 / 背面 / 正面 的 cell_acc（没有 = nan）
    cube = labels.cube
    _, ends = cube.transforms(ep["actions"])
    tex = torch.as_tensor(np.asarray(ep["textures"]), device=labels.device).long()
    labels.grid = G = tex.shape[-1]
    hid = torch.as_tensor(cube.hidden(ends[ep["num_chunks"] - 4]), device=labels.device).repeat_interleave(G * G)
    sid, cell, frac, shade, _ = labels.trace(ends[chunks])
    inner = (sid >= 0) & ((frac - 0.5).abs() < 0.3).all(-1)           # 只看格子中间（离格线远）
    key = (sid * G + cell[..., 1]) * G + cell[..., 0]
    S, out = tex.numel(), []
    for f in range(len(chunks)):
        m = inner[f]
        k, c, sh = key[f][m], frames[f].reshape(-1, 3)[m].float(), shade[f][m]
        cnt = torch.zeros(S, device=k.device).index_add_(0, k, torch.ones_like(sh))
        col = torch.zeros(S, 3, device=k.device).index_add_(0, k, c)
        shd = torch.zeros(S, device=k.device).index_add_(0, k, sh)
        ok = cnt >= min_px
        col, shd = col[ok] / cnt[ok, None], shd[ok] / cnt[ok]
        pred = ((col[:, None] - shd[:, None, None] * labels.pal[None]) ** 2).sum(-1).argmin(-1)
        right, h = (pred == tex.flatten()[ok]).float(), hid[ok]
        nan = torch.tensor(float("nan"), device=k.device)
        out.append(torch.stack([right.mean(), right[h].mean() if h.any() else nan, right[~h].mean() if (~h).any() else nan]))
    return torch.stack(out)


class RubikEvaluator:
    def __init__(self, moves=(0, 4, 8), num=32, batch=16, steps=20, modes=("tf", "free"), seed=0, videos=0, video_dir=None):
        self.vae = VAE()
        self.scene = RubikScene(2, device="cuda")              # 精确渲染（GT 帧、视频）
        self.labels = RubikScene(2, ssaa=1, device="cuda")     # 每个像素是哪个贴纸的哪一格
        t = TASKS["rubik"]
        self.data = {n: CubeDataset(t["split"].format(n), t["root"]) for n in moves}
        self.num, self.batch, self.steps, self.modes, self.seed = num, batch, steps, modes, seed
        self.videos, self.video_dir = videos, video_dir

    def end_frames(self, z):
        # z: [B, 16, 20, 24, 32]（最后 4 个 chunk 的 latent）-> 这 4 个 chunk 的末帧 [B, 4, H, W, 3]（解码 17 + 20 + 20 + 20 帧）
        return torch.cat([self.vae.decode(z[i:i + 8])[:, [16, 36, 56, 76]] for i in range(0, len(z), 8)])

    def save_video(self, ep, z, mode, n):
        # GT（精确渲染）| 生成；生成那一半顶上一条：绿 = 输入的 GT chunk，红 = 自己生成的，黄 = 评测的 chunk
        import imageio.v2 as imageio
        gt = render_episode(self.scene, ep)
        gen = (self.vae.decode(z[None].float())[0].cpu().numpy() * 255).round().astype(np.uint8)
        K = ep["num_chunks"]
        n_ctx = K - 3 if mode == "tf" else 4
        for f in range(len(gen)):
            c = 0 if f <= 16 else 1 + (f - 17) // 20
            gen[f, :4] = (40, 200, 60) if c < n_ctx else (240, 200, 30) if c >= K - 4 else (220, 40, 40)
        out = Path(self.video_dir) / f"n{n:02d}_ep{ep['id']:03d}_{mode}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(out, np.concatenate([gt, gen], 2), fps=16, quality=9, macro_block_size=1)

    def score(self, eps, K, last4, acc):
        # last4：{协议: [B, 16, 20, 24, 32]}，每个 episode 最后 4 个 chunk 的 latent（生成的或 GT）；分数追加进 acc
        chunks = [K - 4, K - 3, K - 2, K - 1]
        gt = torch.stack([self.scene.render(self.scene.cube.transforms(ep["actions"])[1][chunks], ep["textures"])
                          for ep in eps])                                               # [B, 4, H, W, 3]
        for mode, z in last4.items():
            fr = self.end_frames(z.cuda().float())
            mse = ((fr - gt) ** 2).mean((2, 3, 4)).clamp(min=1e-10)                    # [B, 4]
            psnr = -10 * torch.log10(mse)
            sc = torch.stack([cell_scores(self.labels, ep, chunks, fr[j]) for j, ep in enumerate(eps)])  # [B, 4, 3]
            # scr 只在它是生成出来的时候算：tf 里是 GT；N = 0 时它是开头展示的最后一个 chunk，free 里也是 GT
            gen_scr = mode == "vae" or (mode == "free" and K - 4 >= 4)
            parts = {"disp": slice(1, 4), "last": slice(3, 4)} | ({"scr": slice(0, 1)} if gen_scr else {})
            for part, sl in parts.items():
                for m, v in (("acc", sc[:, sl, 0]), ("hid", sc[:, sl, 1]), ("vis", sc[:, sl, 2]), ("psnr", psnr[:, sl])):
                    acc.setdefault(f"{mode}_{part}_{m}", []).append(v.nanmean(1).cpu())

    @staticmethod
    def collect(n, K, acc, res, per, per_episode):
        for k, v in acc.items():
            v = torch.cat(v).float()
            if math.isnan(v.nanmean().item()):  # scr 帧里没有背面贴纸、last 帧里没有正面贴纸
                continue
            res[f"n{n:02d}_{k}"] = v.nanmean().item()
            if per_episode:
                per[f"n{n:02d}_{k}"] = v.tolist()
        res[f"n{n:02d}_chunks"] = K

    @torch.no_grad()
    def run(self, model, per_episode=False):
        # 现场生成（rollout）再评
        was = model.training
        model.eval()
        res, per = {}, {}
        for n, ds in self.data.items():
            N = min(self.num, len(ds))
            if N == 0:  # 这个 split 还没编码
                continue
            K = ds.eps[0]["num_chunks"]
            bs = self.batch if model.max_ctx is not None else max(1, self.batch * 12 // K)  # KV 全留：长 episode 缩 batch
            acc = {}
            for i in range(0, N, bs):
                idx = list(range(i, min(i + bs, N)))
                b = default_collate([ds[j] for j in idx])
                eps = [ds.eps[j] for j in idx]
                x0, a = b["latent"].cuda(), b["actions"].cuda()
                zs = {"vae": x0}
                for mode in self.modes:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, _, zs[mode] = rollout(model, x0, a, K - 3 if mode == "tf" else 4, self.steps,
                                                 [self.seed * 100000 + ep["id"] for ep in eps], keep=True)
                    if i == 0 and self.videos:
                        for j in range(min(self.videos, len(eps))):
                            self.save_video(eps[j], zs[mode][j], mode, n)
                self.score(eps, K, {m: z[:, :, 5 * (K - 4):] for m, z in zs.items()}, acc)
            self.collect(n, K, acc, res, per, per_episode)
        model.train(was)
        return (res, per) if per_episode else res

    @torch.no_grad()
    def run_saved(self, root, per_episode=False, vae=True):
        # 评 infer.py 存下来的 latent（远端推理拿回来的）：root/nXX/{free,tf}.npy + ids.json；GT latent（vae 上限）从本地数据集读
        root, res, per = Path(root), {}, {}
        for n, ds in self.data.items():
            d = root / f"n{n:02d}"
            if not (d / "ids.json").exists():
                continue
            ids = json.loads((d / "ids.json").read_text())
            pos = {ep["id"]: j for j, ep in enumerate(ds.eps)}
            missing = [i for i in ids if i not in pos]
            if missing:
                raise SystemExit(f"n{n:02d}: episodes {missing[:5]} ... are not in the local dataset")
            saved = {m: np.load(d / f"{m}.npy", mmap_mode="r") for m in self.modes if (d / f"{m}.npy").exists()}
            K = ds.eps[pos[ids[0]]]["num_chunks"]
            E = min(self.num, len(ids))
            acc = {}
            for i in range(0, E, self.batch):
                idx = list(range(i, min(i + self.batch, E)))
                eps = [ds.eps[pos[ids[j]]] for j in idx]
                x0 = default_collate([ds[pos[ids[j]]] for j in idx])["latent"].cuda()
                last4 = {"vae": x0[:, :, 5 * (K - 4):]} if vae else {}
                for m, arr in saved.items():
                    z = torch.from_numpy(np.array(arr[idx[0]:idx[-1] + 1])).cuda().float()
                    last4[m] = z[:, :, 5 * (K - 4):] if m == "free" else z
                    if i == 0 and self.videos:  # tf 只存了最后 4 个 chunk：前面接上 GT 就是整段
                        full = z if m == "free" else torch.cat([x0[:, :, :5 * (K - 4)], z], 2)
                        for j in range(min(self.videos, len(eps))):
                            self.save_video(eps[j], full[j], m, n)
                self.score(eps, K, last4, acc)
            self.collect(n, K, acc, res, per, per_episode)
        return (res, per) if per_episode else res


def main():
    # 本地评测远端推理的结果：python rubik_eval.py --saved <inference/<run>/<ckpt>> --out <run>/eval/<ckpt>_ema.json
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--videos", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-vae", action="store_true")  # 不算 VAE 上限（同一个测试集每次都一样，评一个 job 时算一次就够）
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    done = json.loads((Path(args.saved) / "done.json").read_text())
    out = Path(args.out)
    ev = RubikEvaluator(tuple(done["moves"]), args.num, args.batch, modes=tuple(done["protocols"]), videos=args.videos,
                        video_dir=out.with_suffix(""))
    res, per = ev.run_saved(args.saved, per_episode=True, vae=not args.no_vae)
    for k, v in res.items():
        if k.endswith(("_acc", "_hid", "_vis")):
            print(f"{k}: {v:.4f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "inference": done, "step": done.get("step"), "res": res,
                               "per_episode": per}, indent=1))


if __name__ == "__main__":
    main()
