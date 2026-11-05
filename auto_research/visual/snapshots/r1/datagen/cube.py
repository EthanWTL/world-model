# 旋转方块（visual memory 方向的任务）：方块绕竖直轴转，每个 chunk 转 90°；四个侧面各贴一张每个 episode 独立随机的
# K×K 色块图（8 色调色板，没有任何规律），顶面 / 底面纯色；相机、光照固定。
# 用 torch 光线投射渲染（CPU / GPU 都行，默认 4×4 超采样）：每个像素打到哪个面、面上的 (u, v) 都精确已知，
# 所以评测可以直接把"正对的那个面"按面坐标拉平（rectify），和原始图案逐格比。
#
# 时间轴和 Shell Game 一样：chunk 0 = 17 帧，之后每个 chunk 20 帧（Wan2.1 VAE 下都是 5 个 latent 帧）
#   转动的 chunk：前 16 帧转 90°（余弦缓动），后 4 帧停住 -> 每个 chunk 最后一个 latent 帧都是静止的正对姿态
#   chunk 0 一定静止（HOLD）
# 一个 episode（单个方块，间隔 g 个 chunk）：
#   观察（推理时给 GT）：chunk 0 静止 + chunk 1–3 同一方向转 3 次 -> 四个面都正对过一次，最后正对的就是目标面（均匀随机）
#   离开  chunk 4      ：随机方向转走，目标面转到侧边（刚好看不见）
#   间隔  chunk 5..4+g ：目标面一直在背面角度区间（侧边 / 背面 / 另一侧边）里来回转，每个 chunk 照样转 90°（转速不变）
#   返回  chunk 5+g    ：反着离开的方向转回来，目标面回到正对 -> 在这个 chunk 的最后一帧评测
# 多个方块（--cubes n）：一个 chunk 里每个方块各自转 -1 / 0 / +1 格；间隔里目标方块来回转（gap_mode back）
#   或者停在背面、别的方块转（gap_mode hold，用户的"转别的方块来拉长间隔"）
#   python cube.py --preview 4 --gap 4 --out /tmp/cube_preview          （mp4 + 每个 chunk 末帧的拼图）
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

W, H, FPS = 256, 192, 16
HOLD, ADV, RET = 0, 1, 2  # 单个方块的动作：不动 / 正对的面编号 +1 / -1（转 90°）
CODE = {0: HOLD, 1: ADV, -1: RET}  # 多个方块：动作 = Σ CODE[δ_i]·3^i
PALETTE = np.array([[220, 40, 40], [240, 200, 30], [40, 170, 60], [40, 190, 210],
                    [40, 70, 200], [190, 50, 190], [235, 235, 235], [30, 30, 30]], np.float32) / 255
TOP = np.array([200, 200, 200], np.float32) / 255
BOTTOM = np.array([90, 90, 90], np.float32) / 255
BORDER = np.array([45, 45, 45], np.float32) / 255
BG = np.array([75, 85, 100], np.float32) / 255
BORDER_W = 0.05                  # 侧面四周深色边框的宽度（面坐标）；色块占 [BORDER_W, 1 - BORDER_W]²
LIGHT = np.array([0.3, 0.6, 1.0]) / np.linalg.norm([0.3, 0.6, 1.0])  # 指向光源（相机这一侧、偏上偏右）
AMB, DIF = 0.4, 0.6              # 着色 = AMB + DIF·max(0, n·L)；正对的面 = 0.4 + 0.6·0.83 = 0.90


def chunk_frames(c):
    return 17 if c == 0 else 20


def frame_poses(deltas, start):
    # deltas: [K][n] 每个 chunk 每个方块转几格（0 / +1 / -1），start: [n] chunk 0 正对的面
    # -> [T, n] 每一帧连续的"正对面编号"（转角 = -编号·90°）
    p = np.array(start, np.float64)
    out = []
    for c, d in enumerate(deltas):
        d = np.asarray(d, np.float64)
        if c == 0:
            assert not d.any(), "chunk 0 一定静止"
            out += [p.copy() for _ in range(17)]
        else:
            for u in range(1, 21):
                s = min(u, 16) / 16
                out.append(p + d * (0.5 - 0.5 * math.cos(math.pi * s)))
        p = p + d
    return np.stack(out)


class Scene:
    """相机在 +z 方向抬高 ELEV 度看向原点，n 个方块（半边长 1）沿 x 排开，各自绕竖直轴转"""

    def __init__(self, n_cubes=1, ssaa=4, device="cpu", dist=8.0, elev=15.0, spacing=3.6, margin=0.06):
        self.n, self.ssaa, self.device = n_cubes, ssaa, device
        self.centers = np.array([[(i - (n_cubes - 1) / 2) * spacing, 0, 0] for i in range(n_cubes)])
        e = math.radians(elev)
        self.C = np.array([0, dist * math.sin(e), dist * math.cos(e)])
        self.fwd = -self.C / np.linalg.norm(self.C)
        self.right = np.cross(self.fwd, [0, 1, 0])
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.fwd)
        # 焦距和主点：所有转角下所有方块的轮廓都放进画面（四周留 margin），并且上下居中
        pts = []
        for yaw in np.radians(np.arange(0, 91)):
            R = self.rot(yaw)
            pts += [ctr + R @ np.array([sx, sy, sz]) for ctr in self.centers
                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        X, Y = self.plane(np.array(pts))
        self.xc, self.yc = (X.max() + X.min()) / 2, (Y.max() + Y.min()) / 2
        self.f = min(W / 2 * (1 - margin) / ((X.max() - X.min()) / 2), H / 2 * (1 - margin) / ((Y.max() - Y.min()) / 2))
        # 每个（超采样的）子像素一条射线，世界坐标；相机固定，所以只算一次
        xs = (torch.arange(W * ssaa, dtype=torch.float64) + 0.5) / ssaa
        ys = (torch.arange(H * ssaa, dtype=torch.float64) + 0.5) / ssaa
        X = (xs[None, :] - W / 2) / self.f + self.xc
        Y = -(ys[:, None] - H / 2) / self.f + self.yc
        t = lambda v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
        d = X[..., None] * t(self.right) + Y[..., None] * t(self.up) + t(self.fwd)
        self.dirs = (d / d.norm(dim=-1, keepdim=True)).reshape(-1, 3).float().to(device)  # [R, 3]
        self.shade_front = AMB + DIF * max(0.0, LIGHT[2])

    @staticmethod
    def rot(yaw):
        # 方块局部 -> 世界：绕 +y 转 yaw；侧面 i 的法向 (sin φ_i, 0, cos φ_i)，φ_i = i·90°，所以 yaw = -i·90° 时面 i 正对相机
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def plane(self, P):
        # 世界坐标的点 [N, 3] -> 像平面坐标 (X, Y) = (x_c / z_c, y_c / z_c)
        d = P - self.C
        z = d @ self.fwd
        return d @ self.right / z, d @ self.up / z

    def project(self, P):
        # 世界坐标 [N, 3] -> 像素坐标（连续，像素 j 的中心在 j + 0.5）
        X, Y = self.plane(P)
        return W / 2 + self.f * (X - self.xc), H / 2 - self.f * (Y - self.yc)

    def trace(self, yaw):
        # yaw: [F, n] 每帧每个方块的转角（弧度）-> face [F, R]（0–3 侧面，4 顶，5 底，-1 背景）、cube [F, R]、
        #   uv [F, R, 2]、shade [F, R]（R = 子像素个数）
        yaw = torch.as_tensor(yaw, dtype=torch.float32, device=self.device)
        Fn, R = yaw.shape[0], self.dirs.shape[0]
        dirs = self.dirs
        best = torch.full((Fn, R), float("inf"), device=self.device)
        face = torch.full((Fn, R), -1, dtype=torch.long, device=self.device)
        cube = torch.zeros((Fn, R), dtype=torch.long, device=self.device)
        uv = torch.zeros((Fn, R, 2), device=self.device)
        nrm = torch.zeros((Fn, R, 3), device=self.device)
        for i in range(self.n):
            c, s = yaw[:, i].cos()[:, None], yaw[:, i].sin()[:, None]               # [F, 1]
            o = torch.tensor(self.C - self.centers[i], dtype=torch.float32, device=self.device)
            # 世界 -> 局部（R_y(-yaw)）：x' = c·x - s·z，z' = s·x + c·z
            ox, oy, oz = c * o[0] - s * o[2], o[1].expand(Fn, 1), s * o[0] + c * o[2]   # [F, 1]
            dx, dy, dz = c * dirs[:, 0] - s * dirs[:, 2], dirs[:, 1].expand(Fn, R), s * dirs[:, 0] + c * dirs[:, 2]
            O, D = torch.stack([ox.expand(Fn, R), oy.expand(Fn, R), oz.expand(Fn, R)], -1), torch.stack([dx, dy, dz], -1)
            inv = 1 / torch.where(D.abs() < 1e-12, torch.full_like(D, 1e-12), D)
            t1, t2 = (-1 - O) * inv, (1 - O) * inv
            tn, ax = torch.minimum(t1, t2).max(-1)                                     # 进入盒子的 t 和是哪个轴
            tf = torch.maximum(t1, t2).min(-1).values
            hit = (tn < tf) & (tn > 0) & (tn < best)
            P = O + tn[..., None] * D                                                  # 局部坐标的交点
            sign = -torch.gather(D, -1, ax[..., None])[..., 0].sign()                  # 打到的面的法向朝 ±轴
            # x 轴：+ -> 面 1，- -> 面 3；z 轴：+ -> 面 0，- -> 面 2；y 轴：+ -> 顶 4，- -> 底 5
            fid = torch.where(ax == 0, torch.where(sign > 0, 1, 3), torch.where(ax == 2, torch.where(sign > 0, 0, 2),
                                                                                    torch.where(sign > 0, 4, 5)))
            # 侧面的 (u, v)：u 沿面的"右"方向 t_i = (cos φ_i, 0, -sin φ_i)，v 从上到下
            phi = fid.float() * (math.pi / 2)
            u = (P[..., 0] * phi.cos() - P[..., 2] * phi.sin() + 1) / 2
            v = (1 - P[..., 1]) / 2
            # 世界坐标的法向：侧面 (sin(φ+yaw), 0, cos(φ+yaw))，顶 / 底 (0, ±1, 0)
            side = fid < 4
            n_w = torch.stack([torch.where(side, (phi + yaw[:, i:i + 1]).sin(), torch.zeros_like(phi)),
                               torch.where(side, torch.zeros_like(phi), torch.where(fid == 4, 1.0, -1.0)),
                               torch.where(side, (phi + yaw[:, i:i + 1]).cos(), torch.zeros_like(phi))], -1)
            best = torch.where(hit, tn, best)
            face = torch.where(hit, fid, face)
            cube = torch.where(hit, i, cube)
            uv = torch.where(hit[..., None], torch.stack([u, v], -1), uv)
            nrm = torch.where(hit[..., None], n_w, nrm)
        light = torch.tensor(LIGHT, dtype=torch.float32, device=self.device)
        shade = AMB + DIF * (nrm @ light).clamp(min=0)
        return face, cube, uv, shade

    def render(self, yaw, tex):
        # yaw: [F, n]；tex: [n, 4, K, K] 调色板编号 -> [F, H, W, 3] float [0, 1]
        face, cube, uv, shade = self.trace(yaw)
        tex = torch.as_tensor(np.asarray(tex), device=self.device).long()
        K = tex.shape[-1]
        pal = torch.tensor(PALETTE, device=self.device)
        const = torch.tensor(np.stack([TOP, BOTTOM, BG, BORDER]), device=self.device)
        g = ((uv - BORDER_W) / (1 - 2 * BORDER_W) * K).floor().long()              # 第几列 / 第几行
        inside = (g >= 0).all(-1) & (g < K).all(-1)
        g = g.clamp(0, K - 1)
        idx = tex[cube, face.clamp(0, 3), g[..., 1], g[..., 0]]                       # [F, R]
        col = pal[idx]
        col = torch.where((face < 4)[..., None] & ~inside[..., None], const[3], col)  # 边框
        col = torch.where((face == 4)[..., None], const[0], col)
        col = torch.where((face == 5)[..., None], const[1], col)
        col = col * shade[..., None]
        col = torch.where((face < 0)[..., None], const[2], col)
        s = self.ssaa
        img = col.view(-1, H * s, W * s, 3).permute(0, 3, 1, 2)
        return F.avg_pool2d(img, s).permute(0, 2, 3, 1).clamp(0, 1)

    def face_grid(self, cube=0, size=64):
        # 任何一个侧面正对相机时（几何都一样），它的色块区域 uv ∈ [BORDER_W, 1 - BORDER_W]² 在画面里的位置：
        #   size × size 个采样点（第 r 行第 c 列 = 面上从上往下第 r 个、从左往右第 c 个）-> grid_sample 的坐标 [size, size, 2]
        a = (np.arange(size) + 0.5) / size * (1 - 2 * BORDER_W) + BORDER_W
        uu, vv = np.meshgrid(a, a)
        P = self.centers[cube] + np.stack([2 * uu - 1, 1 - 2 * vv, np.ones_like(uu)], -1).reshape(-1, 3)
        x, y = self.project(P)
        return torch.tensor(np.stack([2 * x / W - 1, 2 * y / H - 1], -1).reshape(size, size, 2), dtype=torch.float32)

    @torch.no_grad()
    def silhouettes(self, offsets_deg, cube=0):
        # 方块 cube 从正对转 offset 度时的轮廓（其余方块不画）-> [len(offsets), H, W] 覆盖率 [0, 1]
        yaw = np.zeros((len(offsets_deg), self.n))
        yaw[:, cube] = np.radians(offsets_deg)
        face, c, _, _ = self.trace(yaw)
        m = ((face >= 0) & (c == cube)).float().view(-1, 1, H * self.ssaa, W * self.ssaa)
        return F.avg_pool2d(m, self.ssaa)[:, 0]


def closed_walk(rng, g):
    # 背面区间 {1, 2, 3}（1 / 3 = 目标面在两侧刚好看不见，2 = 在背面）里从 1 出发、g 步后回到 1 的游走，
    # 在所有这样的路径里均匀抽（先数每个位置还剩 k 步能回到 1 的路径数）
    cnt = [[1.0 if r == 1 else 0.0 for r in range(4)]]
    for k in range(1, g + 1):
        cnt.append([0.0] + [sum(cnt[k - 1][q] for q in (r - 1, r + 1) if 1 <= q <= 3) for r in (1, 2, 3)])
    path = [1]
    for k in range(g, 0, -1):
        nb = [q for q in (path[-1] - 1, path[-1] + 1) if 1 <= q <= 3]
        w = np.array([cnt[k - 1][q] for q in nb])
        path.append(nb[rng.choice(len(nb), p=w / w.sum())])
    assert path[-1] == 1
    return path


def episode(rng, gap, grid=4, colors=8, n_cubes=1, gap_mode="back"):
    # 抽一个 episode（全部是 int / list，可以直接 json）
    assert gap >= 0 and (gap % 2 == 0 or gap_mode == "hold"), "来回转的间隔只能是偶数个 chunk"
    assert gap_mode in ("back", "hold") and (gap_mode == "back" or n_cubes > 1), "hold 要有别的方块在转"
    n = n_cubes
    tex = rng.integers(colors, size=(n, 4, grid, grid))
    tc, tf = int(rng.integers(n)), int(rng.integers(4))           # 目标方块、目标面
    d_obs = rng.choice([-1, 1], size=n)
    end = rng.integers(4, size=n)                                  # 观察结束时每个方块正对的面
    end[tc] = tf
    start = (end - 3 * d_obs) % 4
    deltas = [[0] * n] + [d_obs.tolist()] * 3

    def others(moving):
        # 非目标方块这个 chunk 各转几格；moving：至少有一个在转（hold 的间隔里靠它们拉长时间）
        while True:
            d = rng.integers(-1, 2, size=n)
            d[tc] = 0
            if n == 1 or not moving or d.any():
                return d

    d_l = int(rng.choice([-1, 1]))
    d = others(False)
    d[tc] = d_l
    deltas.append(d.tolist())
    if gap_mode == "back":
        walk = closed_walk(rng, gap)
        for r0, r1 in zip(walk, walk[1:]):
            d = others(False)
            d[tc] = (r1 - r0) * d_l                                    # 离开的方向 = 离目标面更远
            deltas.append(d.tolist())
    else:
        for _ in range(gap):
            deltas.append(others(True).tolist())
    d = others(False)
    d[tc] = -d_l
    deltas.append(d.tolist())
    pose = (start[None] + np.cumsum(deltas, 0)) % 4                     # [K, n] 每个 chunk 结束时正对的面
    actions = [int(sum(CODE[int(x)] * 3 ** i for i, x in enumerate(row))) for row in deltas]
    # 目标面每一帧看不看得见：和正对差的角度 ∈ (-90°, 90°)
    rel = (frame_poses(deltas, start)[:, tc] - tf) % 4
    vis = (rel < 1 - 1e-6) | (rel > 3 + 1e-6)
    K = len(deltas)
    ret = K - 1
    first = 17 + 20 * (ret - 1)                                        # 返回 chunk 的第一帧
    gap_frames = 0
    while not vis[first - 1 - gap_frames]:
        gap_frames += 1
    assert pose[ret, tc] == tf and pose[4, tc] != tf
    return {"n_cubes": n, "grid": grid, "colors": colors, "gap_mode": gap_mode, "textures": tex.tolist(),
            "start": start.tolist(), "deltas": deltas, "actions": actions, "pose": pose.tolist(),
            "target_cube": tc, "target": tf, "obs": 4, "leave": 4, "ret": ret, "gap": gap,
            "gap_frames": gap_frames, "writes_gap": gap, "num_chunks": K, "num_frames": 17 + 20 * (K - 1),
            "target_visible": [bool(vis[(0 if c == 0 else 17 + 20 * (c - 1)):17 + 20 * c].any()) for c in range(K)]}


@torch.no_grad()
def render_episode(scene, ep, batch=24):
    # -> uint8 [T, H, W, 3]
    yaw = -frame_poses(ep["deltas"], ep["start"]) * (math.pi / 2)
    out = [scene.render(yaw[i:i + batch], ep["textures"]) for i in range(0, len(yaw), batch)]
    return (torch.cat(out) * 255).round().byte().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=int, default=4)
    ap.add_argument("--gap", type=int, default=4)
    ap.add_argument("--cubes", type=int, default=1)
    ap.add_argument("--gap-mode", default="back", choices=["back", "hold"])
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import imageio.v2 as imageio
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = Scene(args.cubes, device=args.device)
    rng = np.random.default_rng(args.seed)
    for i in range(args.preview):
        ep = episode(rng, args.gap, args.grid, n_cubes=args.cubes, gap_mode=args.gap_mode)
        fr = render_episode(scene, ep)
        imageio.mimwrite(out / f"ep{i}_g{args.gap}.mp4", fr, fps=FPS, quality=9, macro_block_size=1)
        ends = [16 + 20 * c for c in range(ep["num_chunks"])]                # 每个 chunk 的最后一帧
        imageio.imwrite(out / f"ep{i}_g{args.gap}_ends.png", np.concatenate([fr[e] for e in ends], 1))
        (out / f"ep{i}_g{args.gap}.json").write_text(json.dumps(ep))
        print(i, "actions", ep["actions"], "pose", [p[0] for p in ep["pose"]], "target", ep["target"],
              "gap_frames", ep["gap_frames"], "visible", ep["target_visible"])


if __name__ == "__main__":
    main()
