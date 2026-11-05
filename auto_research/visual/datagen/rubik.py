# 魔方（visual memory 方向的第二个任务，用户 2026-11-03："把任务变成魔方 … 每一个小面上的图案都是随机出来的纹理 …
#   动作就是跟魔方操作一样 … 最后展示就是固定的6个面的展示 几个整体转的额外动作"）
# 2×2 魔方（代码对 n 阶通用）：每个贴纸是一张独立随机的 g×g 色块图（8 色调色板，没有任何规律），塑料是黑的；
# 相机斜着看（方位 45°、仰角 35°，看到上 / 前 / 右三个面），光照固定，背景灰蓝。
# 一个 episode（用户选的"展示 -> 打乱 -> 展示"）：
#   chunk 0       ：不动（17 帧）
#   chunk 1–3     ：整体转 y y x（开头展示：后 / 左 / 下依次转到看得见的位置 -> 所有贴纸都看过一遍）
#   chunk 4..3+N  ：N 步随机拧面（12 种里均匀抽，不紧接着拧回上一步）
#   最后 3 个     ：同一套整体转 y y x（最后展示）；评测 = 打乱最后一帧 + 这 3 个 chunk 的末帧（都静止），逐个贴纸逐格比
# 动作 19 个：0 不动；1–12 拧面 U U' D D' L L' R R' F F' B B'（按当前位置命名：U = 现在朝上的那层，从那面外面看顺时针）；
#   13–18 整体转 x x' y y' z z'（x / y / z 和 R / U / F 同方向）
# 时间轴和旋转方块一样：chunk 0 = 17 帧，之后每个 chunk 20 帧 = 16 帧转 90°（余弦缓动）+ 4 帧停住（Wan VAE 下都是 5 个 latent 帧）
# 渲染：torch 光线投射。每个小块（cubie）是一个边长 1 的盒子，带自己的累计旋转；拧的那一层这一帧再绕轴多转一个角度。
#   打到的面在小块自己的（初始）坐标系里判断：朝外的面 = 贴纸（贴纸坐标系跟着小块一起转），朝里的面和贴纸四周 = 黑塑料
#   小样（渲染 + VAE 往返 + 预览视频）：datagen/rubik_sample.py
import math

import numpy as np
import torch
import torch.nn.functional as F

W, H, FPS = 256, 192, 16
PALETTE = np.array([[220, 40, 40], [240, 200, 30], [40, 170, 60], [40, 190, 210],
                    [40, 70, 200], [190, 50, 190], [235, 235, 235], [245, 130, 30]], np.float32) / 255  # 没有黑：黑 = 塑料
PLASTIC = np.array([22, 22, 22], np.float32) / 255
BG = np.array([75, 85, 100], np.float32) / 255
BORDER_W = 0.07                  # 贴纸四周黑边（小块面坐标，边长 1）
LIGHT = np.array([0.25, 1.0, 0.55]) / np.linalg.norm([0.25, 1.0, 0.55])  # 上面最亮，前面次之，右面最暗
AMB, DIF = 0.55, 0.45
FACES = "UDLRFB"
AXIS = {"U": (1, 1), "D": (1, -1), "L": (0, -1), "R": (0, 1), "F": (2, 1), "B": (2, -1)}  # 面 -> (轴, 朝向)
# 每个面贴纸的 (u, v) 方向（从外面看：u 向右，v 向下；上面 = 后在上、前在下，下面 = 前在上）
UVAX = {"U": ((1, 0, 0), (0, 0, 1)), "D": ((1, 0, 0), (0, 0, -1)), "L": ((0, 0, 1), (0, -1, 0)),
        "R": ((0, 0, -1), (0, -1, 0)), "F": ((1, 0, 0), (0, -1, 0)), "B": ((-1, 0, 0), (0, -1, 0))}
ACTIONS = ["hold"] + [f + s for f in FACES for s in ("", "'")] + [r + s for r in "xyz" for s in ("", "'")]
TURNS = list(range(1, 13))       # 拧面的动作编号
DISPLAY = [ACTIONS.index(a) for a in ("y", "y", "x")]


def rot(axis, c, s):
    # 绕 +axis 转（右手）：cos、sin 给定
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def spec(a):
    # 动作编号 -> None（不动）或 (轴, 绕 +轴转几个 90°, 层)；层 = (轴, 朝向) 只拧最外层，None = 整个魔方
    name = ACTIONS[a]
    if name == "hold":
        return None
    prime = name.endswith("'")
    if name[0] in AXIS:  # 从那一面外面看顺时针 = 绕 (朝向·轴) 转 -90° = 绕 +轴 转 -朝向·90°
        ax, sd = AXIS[name[0]]
        return ax, (sd if prime else -sd), (ax, sd)
    return "xyz".index(name[0]), (1 if prime else -1), None


def inverse(a):
    return a + 1 if a % 2 == 1 else a - 1  # U <-> U'，x <-> x' …（1–18 两两一对）


class Cube:
    """n 阶魔方的小块和贴纸（初始 = home 坐标系，魔方中心在原点，小块边长 1）"""

    def __init__(self, n=2):
        self.n, k = n, (n - 1) / 2
        cs = [i - k for i in range(n)]
        self.home = np.array([(x, y, z) for x in cs for y in cs for z in cs if max(abs(x), abs(y), abs(z)) == k])
        self.sticker = -np.ones((len(self.home), 3, 2), int)  # [小块, 轴, 朝向 (0 = -, 1 = +)] -> 贴纸编号，-1 = 朝里
        self.face_of, self.info, sid = [], [], 0  # info[贴纸] = (小块, 轴, 朝向 ±1)：home 坐标系里贴在哪个小块的哪个面
        for f in FACES:  # 按面 U D L R F B，面里从上到下、从左到右编号
            ax, sd = AXIS[f]
            u, v = (np.array(x) for x in UVAX[f])
            on = [i for i in range(len(self.home)) if self.home[i, ax] * sd == k]
            for i in sorted(on, key=lambda i: (self.home[i] @ v, self.home[i] @ u)):
                self.sticker[i, ax, (sd + 1) // 2] = sid
                self.face_of.append(f)
                self.info.append((i, ax, sd))
                sid += 1
        self.num_stickers = sid
        self.uvax = np.zeros((3, 2, 2, 3))  # [轴, 朝向, u / v, 3]
        for f, (ax, sd) in AXIS.items():
            self.uvax[ax, (sd + 1) // 2] = UVAX[f]

    def hidden(self, M):
        # M [C, 3, 3]（某个 chunk 结束时的整数旋转）-> [S] bool：贴纸这时朝 -x / -y / -z（左 / 下 / 后，相机看不见）
        n = np.array([M[c] @ (sd * np.eye(3, dtype=np.int64)[ax]) for c, ax, sd in self.info])
        return n.sum(1) < 0

    def transforms(self, actions):
        # -> A [T, C, 3, 3]：每帧每个小块 home -> 世界 的旋转；ends [K, C, 3, 3]：每个 chunk 结束时（整数）
        C, k = len(self.home), (self.n - 1) / 2
        M = np.repeat(np.eye(3, dtype=np.int64)[None], C, 0)
        out, ends = [], []
        for c, a in enumerate(actions):
            sp = spec(a)
            if c == 0 or sp is None:
                assert c > 0 or a == 0, "chunk 0 一定不动"
                out += [M.astype(np.float32)] * (17 if c == 0 else 20)
            else:
                ax, q, layer = sp
                cur = np.einsum("cij,cj->ci", M, self.home)
                sel = np.ones(C, bool) if layer is None else cur[:, layer[0]] * layer[1] > k - 0.5
                for u in range(1, 21):
                    th = q * (0.5 - 0.5 * math.cos(math.pi * min(u, 16) / 16)) * math.pi / 2
                    A = M.astype(np.float32)
                    A[sel] = rot(ax, math.cos(th), math.sin(th)) @ M[sel]
                    out.append(A)
                M = M.copy()
                M[sel] = rot(ax, round(math.cos(q * math.pi / 2)), round(math.sin(q * math.pi / 2))) @ M[sel]
            ends.append(M.copy())
        return np.stack(out), np.stack(ends)


class RubikScene:
    """相机在 (方位 az, 仰角 el) 方向、距离 dist 看向原点；画面放得下魔方的外接球（拧到一半也不出画）"""

    def __init__(self, n=2, ssaa=4, device="cpu", dist=9.0, az=45.0, el=35.0, margin=0.04):
        self.cube, self.ssaa, self.device = Cube(n), ssaa, device
        a, e = math.radians(az), math.radians(el)
        self.C = dist * np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
        self.fwd = -self.C / np.linalg.norm(self.C)
        self.right = np.cross(self.fwd, [0, 1, 0])
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.fwd)
        g = np.arange(4000) + 0.5  # 外接球面上的点（Fibonacci）
        phi, th = np.arccos(1 - 2 * g / 4000), math.pi * (1 + 5 ** 0.5) * g
        P = math.sqrt(3) * n / 2 * np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], -1)
        X, Y = self.plane(P)
        self.xc, self.yc = (X.max() + X.min()) / 2, (Y.max() + Y.min()) / 2
        self.f = min(W / 2 * (1 - margin) / ((X.max() - X.min()) / 2), H / 2 * (1 - margin) / ((Y.max() - Y.min()) / 2))
        xs = (torch.arange(W * ssaa, dtype=torch.float64) + 0.5) / ssaa
        ys = (torch.arange(H * ssaa, dtype=torch.float64) + 0.5) / ssaa
        X = (xs[None, :] - W / 2) / self.f + self.xc
        Y = -(ys[:, None] - H / 2) / self.f + self.yc
        t = lambda v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
        d = X[..., None] * t(self.right) + Y[..., None] * t(self.up) + t(self.fwd)
        self.dirs = (d / d.norm(dim=-1, keepdim=True)).reshape(-1, 3).float().to(device)  # [R, 3]
        dev = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=device)  # noqa: E731
        self.home, self.sticker_t = dev(self.cube.home), dev(self.cube.sticker, torch.long)
        self.uvax, self.light, self.cam = dev(self.cube.uvax), dev(LIGHT), dev(self.C)
        self.pal, self.plastic, self.bg = dev(PALETTE), dev(PLASTIC), dev(BG)

    def plane(self, P):
        d = P - self.C
        z = d @ self.fwd
        return d @ self.right / z, d @ self.up / z

    def trace(self, A):
        # A: [F, C, 3, 3] -> sid [F, R]（贴纸编号，-1 = 塑料 / 背景）、cell [F, R, 2]（列, 行）、frac [F, R, 2]（格子里的位置）、
        #   shade [F, R]、hit [F, R]
        A = torch.as_tensor(A, dtype=torch.float32, device=self.device)
        Fn, C = A.shape[:2]
        R, dirs = self.dirs.shape[0], self.dirs
        best = torch.full((Fn, R), float("inf"), device=self.device)
        cid = torch.full((Fn, R), -1, dtype=torch.long, device=self.device)
        P = torch.zeros((Fn, R, 3), device=self.device)
        axh = torch.zeros((Fn, R), dtype=torch.long, device=self.device)
        sgn = torch.zeros((Fn, R), device=self.device)
        for i in range(C):
            Ai = A[:, i]                                              # [F, 3, 3]
            oh = torch.einsum("fji,j->fi", Ai, self.cam)[:, None]    # 相机在小块 home 坐标系里 [F, 1, 3]
            dh = torch.einsum("rj,fji->fri", dirs, Ai)               # 射线方向 [F, R, 3]
            inv = 1 / torch.where(dh.abs() < 1e-9, torch.full_like(dh, 1e-9), dh)
            t1, t2 = (self.home[i] - 0.5 - oh) * inv, (self.home[i] + 0.5 - oh) * inv
            tn, ax = torch.minimum(t1, t2).max(-1)
            tf = torch.maximum(t1, t2).min(-1).values
            hit = (tn < tf) & (tn > 0) & (tn < best)
            ph = oh + tn[..., None] * dh - self.home[i]              # 小块中心为原点的交点
            s = -torch.gather(dh, -1, ax[..., None])[..., 0].sign()  # 打到的面朝 ±轴
            best = torch.where(hit, tn, best)
            cid = torch.where(hit, i, cid)
            P = torch.where(hit[..., None], ph, P)
            axh = torch.where(hit, ax, axh)
            sgn = torch.where(hit, s, sgn)
        hit = cid >= 0
        side = (sgn > 0).long()
        sid = torch.where(hit, self.sticker_t[cid.clamp(min=0), axh, side], -1)
        uv = (self.uvax[axh, side] * P[..., None, :]).sum(-1) + 0.5          # [F, R, 2]
        x = (uv - BORDER_W) / (1 - 2 * BORDER_W)
        inside = (x >= 0).all(-1) & (x < 1).all(-1)
        sid = torch.where(inside, sid, -1)
        G = self.grid
        cell = (x * G).floor().long().clamp(0, G - 1)
        frac = x * G - cell
        nw = A.transpose(-1, -2)[torch.arange(Fn, device=self.device)[:, None], cid.clamp(min=0), axh] * sgn[..., None]
        shade = AMB + DIF * (nw @ self.light).clamp(min=0)
        return sid, cell, frac, shade, hit

    def render(self, A, tex):
        # tex: [S, g, g] 调色板编号 -> [F, H, W, 3] float [0, 1]
        tex = torch.as_tensor(np.asarray(tex), device=self.device).long()
        self.grid = tex.shape[-1]
        sid, cell, _, shade, hit = self.trace(A)
        col = self.pal[tex[sid.clamp(min=0), cell[..., 1], cell[..., 0]]]
        col = torch.where((sid >= 0)[..., None], col, self.plastic)
        col = torch.where(hit[..., None], col * shade[..., None], self.bg)
        s = self.ssaa
        img = col.view(-1, H * s, W * s, 3).permute(0, 3, 1, 2)
        return F.avg_pool2d(img, s).permute(0, 2, 3, 1).clamp(0, 1)


@torch.no_grad()
def render_episode(scene, ep, batch=16):
    A, _ = scene.cube.transforms(ep["actions"])
    out = [scene.render(A[i:i + batch], ep["textures"]) for i in range(0, len(A), batch)]
    return (torch.cat(out) * 255).round().byte().cpu().numpy()


def episode(rng, n_moves, n=2, grid=3, colors=8):
    # 抽一个 episode（全部 int / list，可以直接 json）
    tex = rng.integers(colors, size=(6 * n * n, grid, grid))
    moves = []
    for _ in range(n_moves):
        while True:
            m = int(rng.choice(TURNS))
            if not moves or m != inverse(moves[-1]):
                break
        moves.append(m)
    actions = [0] + DISPLAY + moves + DISPLAY
    K = len(actions)
    return {"n": n, "grid": grid, "colors": colors, "textures": tex.tolist(), "actions": actions,
            "action_names": [ACTIONS[a] for a in actions], "moves": n_moves, "num_chunks": K,
            "num_frames": 17 + 20 * (K - 1), "display": [1, 2, 3], "scramble": list(range(4, 4 + n_moves)),
            "final": [K - 3, K - 2, K - 1], "eval_chunks": [K - 4, K - 3, K - 2, K - 1]}
