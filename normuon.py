import torch

# Polar Express 系数（Dion 的 NorMuon 默认用它）：每一步把每个奇异值 s 变成 a·s + b·s³ + c·s⁵
COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def orthogonalize(G):
    # G: [out, in] = U S Vᵀ  ->  ≈ U Vᵀ：方向不变，所有奇异值拉到 ≈ 1
    X = G.bfloat16()
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.mT                                # 统一成宽矩阵，X @ X.mT 就是小的那一边
    X = X / (X.norm() * 1.02 + 1e-6)            # Frobenius 范数 ≥ 最大奇异值，除完之后所有奇异值 < 1
    for a, b, c in COEFFS:
        A = X @ X.mT                            # = U S² Uᵀ
        X = a * X + (b * A + c * A @ A) @ X     # = U (a·S + b·S³ + c·S⁵) Vᵀ
    return X.mT if tall else X

class NorMuon(torch.optim.Optimizer):
    # 一个优化器管两组参数：Linear 的权重矩阵走 NorMuon，其余（bias、norm、modulation、embedding、Conv3d）走 AdamW
    # 超参照论文 D.2 + Dion 的默认值；两组的 lr 都是 η，矩阵的步子在 normuon() 里再乘 0.2·√max(out, in)
    def __init__(self, matrices, others, lr=2e-4, momentum=0.95, beta2=0.95, betas=(0.9, 0.95),
                 weight_decay=0.01, eps=1e-8):
        groups = [dict(params=list(matrices), muon=True), dict(params=list(others), muon=False)]
        super().__init__(groups, dict(lr=lr, momentum=momentum, beta2=beta2, betas=betas,
                                      weight_decay=weight_decay, eps=eps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                p.mul_(1 - g["lr"] * g["weight_decay"])  # weight decay：两组一样，先把参数缩一点
                if g["muon"]:
                    self.normuon(p, g)
                else:
                    self.adamw(p, g)

    def normuon(self, p, g):
        s = self.state[p]  # 这个参数自己的状态；state_dict() 会存下来，续训读回来
        if not s:
            s["m"] = torch.zeros_like(p)         # 动量 [out, in]
            s["v"] = torch.zeros_like(p[:, :1])  # 每个输出神经元（每一行）一个二阶矩 [out, 1]
        s["m"].mul_(g["momentum"]).add_(p.grad)  # M = μ·M + G
        U = orthogonalize(s["m"]).float()        # 所有奇异值 ≈ 1
        norm = U.norm()
        s["v"].lerp_(U.square().mean(dim=1, keepdim=True), 1 - g["beta2"])  # 每一行更新量的平方均值，做滑动平均
        U = U / (s["v"].sqrt() + g["eps"])       # 每一行除以自己的 RMS：每个神经元的步子一样大
        U = U * (norm / U.norm().clamp(min=1e-8))  # 总范数还原成正交化之后的大小
        p.add_(U, alpha=-g["lr"] * 0.2 * max(p.shape) ** 0.5)  # 更新的 RMS ≈ 0.2·lr，和 AdamW 差不多

    def adamw(self, p, g):
        s = self.state[p]
        if not s:
            s["step"] = 0
            s["m"] = torch.zeros_like(p)  # 一阶矩
            s["v"] = torch.zeros_like(p)  # 二阶矩
        s["step"] += 1
        b1, b2 = g["betas"]
        s["m"].lerp_(p.grad, 1 - b1)           # m = b1·m + (1 - b1)·g
        s["v"].lerp_(p.grad.square(), 1 - b2)  # v = b2·v + (1 - b2)·g²
        denom = (s["v"] / (1 - b2 ** s["step"])).sqrt() + g["eps"]  # 除掉偏差（刚开始 m、v 都偏小）
        p.addcdiv_(s["m"], denom, value=-g["lr"] / (1 - b1 ** s["step"]))