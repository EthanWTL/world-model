import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def block_mask(rows, cols, heads):
    return torch.block_diag(*[torch.ones(rows // heads, cols // heads)] * heads)


class ProjMemory(nn.Module):
    def __init__(self, dim=1152, nf=288, heads=4, loss='delta', eta0=1e-3, renorm=True):
        super().__init__()
        assert loss in ('delta','dot'), loss
        self.dim, self.nf, self.loss, self.renorm = dim, nf, loss, renorm
        self.target = nn.Linear(dim, nf, bias = False)
        nn.init.zeros_(self.target.weight)
        self.eta = nn.Linear(dim, 1)
        nn.init.zeros_(self.eta.weight)
        nn.init.constant_(self.eta.bias, math.log(math.expm1(eta0)))
        self.register_buffer("mask", block_mask(nf, dim, heads), persistent=False)

    def init(self, B, device):
        return torch.zeros(B, self.nf, self.dim, device = device)

    def read(self, S, x):
        return F.pad(x @ S.mT, (self.dim - self.nf, 0))

    def write(self, S, x, W):
        #x[B, 5*12*16, 1152]
        tgt = self.target(x).float() #[B, 5*12*16, 288]
        eta = F.softplus(self.eta(x).float()) #[B, 960, 1] learning rate
        if self.loss == 'delta':
            err = (x @ S.mT).float() - tgt # S[B, 288, 1152]
            #err [B, 960, 288]
        else:
            err = -tgt
        G = (err * eta / x.shape[1]).mT @ x
        S = S - G * self.mask
        if self.renorm:  # W_fast + ΔW 每一行的长度拉回 slow 的：方向可以变，长度不变
            Ws = W[-self.nf:]                       # [nf, D] fast 通道的 slow weight
            Wf = Ws + S                             # [B, nf, D] 这个样本此刻真正在用的 fast 权重
            S = Wf * (Ws.norm(dim=-1, keepdim=True) / Wf.norm(dim=-1, keepdim=True).clamp_min(1e-6)) - Ws
        return S