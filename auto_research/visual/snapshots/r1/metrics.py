# 回访评测的指标：把正对相机的那个面按面坐标拉平（rectify，只取色块区域 = 面的内部 ROI，不含边框），和 GT 比
#   mse / psnr / ssim / lpips：拉平后的 [3, G, G] 图和 GT 渲染图比
#   cell_acc：每一格中间一半区域的平均颜色 -> 最近的调色板颜色（正对时的着色），和这个面的原始图案比（瞎猜 = 1/8）
#   ident：生成的面和 4 个面的原始图案各自对上几格，最像的是不是目标面；wrong_face = 最像的是别的面（转错了 = 姿态错误）
#   yaw_err：画面里方块的轮廓（和背景色差得远的像素）和"从正对转 δ 度"的轮廓比 IoU，最好的 |δ|（几何上的姿态错误）
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "datagen"))
from cube import BG, PALETTE, Scene  # noqa: E402


def gaussian_ssim(x, y, win=11, sigma=1.5):
    # x, y: [B, 3, G, G] in [0, 1] -> [B]（标准 SSIM：高斯窗、valid 卷积、每个通道算完取平均）
    g = torch.exp(-(torch.arange(win, device=x.device) - win // 2).float() ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())[:, None] * (g / g.sum())[None, :]
    k = g.expand(x.shape[1], 1, win, win)
    blur = lambda t: F.conv2d(t, k, groups=x.shape[1])  # noqa: E731
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.mean((1, 2, 3))


class FaceMetrics:
    def __init__(self, scene=None, grid=4, size=64, device="cuda", use_lpips=True, cube=0):
        self.scene = scene or Scene(1, device="cpu")
        self.grid, self.size, self.device = grid, size, device
        self.xy = self.scene.face_grid(cube, size).to(device)                                   # [G, G, 2]
        self.pal = torch.tensor(PALETTE * self.scene.shade_front, device=device)                # 正对时看到的 8 种颜色
        self.bg = torch.tensor(BG, device=device)
        self.offsets = np.arange(-45, 46)
        self.sil = self.scene.silhouettes(self.offsets, cube).to(device) > 0.5                  # [91, H, W]
        self.lpips = None
        if use_lpips:
            import lpips
            self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def rectify(self, frames):
        # frames: [B, H, W, 3] float [0, 1] -> [B, 3, G, G]
        x = frames.permute(0, 3, 1, 2).float().to(self.device)
        return F.grid_sample(x, self.xy.expand(x.shape[0], -1, -1, -1), mode="bilinear", align_corners=False)

    def cells(self, rect):
        # [B, 3, G, G] -> [B, K, K] 每一格中间一半区域的平均颜色最近的调色板编号
        K, s = self.grid, self.size // self.grid
        c = rect.unflatten(2, (K, s)).unflatten(4, (K, s))[:, :, :, s // 4:s - s // 4, :, s // 4:s - s // 4]
        mean = c.mean((3, 5)).permute(0, 2, 3, 1)                                              # [B, K, K, 3]
        return (mean[..., None, :] - self.pal).norm(dim=-1).argmin(-1)

    def yaw_error(self, frames):
        # 方块轮廓 vs 正对转 δ 度的轮廓：IoU 最大的 |δ|（度），以及 δ = 0 的 IoU
        m = ((frames.float().to(self.device) - self.bg).norm(dim=-1) > 0.15)                   # [B, H, W]
        inter = (m[:, None] & self.sil[None]).sum((2, 3)).float()
        union = (m[:, None] | self.sil[None]).sum((2, 3)).float().clamp(min=1)
        iou = inter / union                                                                     # [B, 91]
        return torch.tensor(np.abs(self.offsets), device=self.device)[iou.argmax(1)].float(), iou[:, 45]

    @torch.no_grad()
    def __call__(self, gen, gt, faces, target):
        # gen / gt: [B, H, W, 3] float [0, 1]（gt = 精确渲染的那一帧）；faces: [B, 4, K, K] 这个方块 4 个面的图案；target: [B]
        # -> {指标: [B] tensor}
        rg, rt = self.rectify(gen), self.rectify(gt)
        mse = (rg - rt).pow(2).mean((1, 2, 3))
        out = {"mse": mse, "psnr": -10 * torch.log10(mse.clamp(min=1e-10)), "ssim": gaussian_ssim(rg, rt)}
        if self.lpips is not None:
            out["lpips"] = self.lpips(rg * 2 - 1, rt * 2 - 1).flatten()
        faces = torch.as_tensor(faces, device=self.device)
        target = torch.as_tensor(target, device=self.device)
        cls = self.cells(rg)
        match = (cls[:, None] == faces).float().mean((2, 3))                                     # [B, 4] 和每个面对上几成
        best = match.argmax(1)
        out["cell_acc"] = match.gather(1, target[:, None])[:, 0]
        out["ident"] = (best == target).float()
        # 姿态错误（转错了面）：画出来的明显是别的面（至少一半的格子对得上别的面；瞎画的话几乎不可能）
        out["wrong_face"] = ((best != target) & (match.max(1).values >= 0.5)).float()
        out["yaw_err"], out["iou0"] = self.yaw_error(gen)
        tgt = faces[torch.arange(len(target), device=self.device), target]                       # [B, K, K]
        out["cell_acc_gt"] = (self.cells(rt) == tgt).float().mean((1, 2))                         # 自检：GT 自己应该 = 1
        return out
