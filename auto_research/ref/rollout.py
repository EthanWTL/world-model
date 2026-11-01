import torch
from diffusion import sigma_schedule

class Rollout:
    def __init__(self, model, max_ctx=None, frames_per_chunk=5):
        self.model, self.max_ctx, self.fpc = model, max_ctx, frames_per_chunk
        self.cache = [{"k": [], "v": []} for _ in model.blocks]  # 每层一份：窗口里每个干净 chunk 的 K/V
        self.start = 0 # 窗口里的干净 chunk [B, C, 5, H, W] 和它们的动作 [B]

    def commit(self, x, a):
        # x: [B, C, 5, H, W] 干净的 chunk；a: [B]。用 σ = 0 过一遍模型（输出不要），每层把它的 K/V 存进 cache
        t = torch.zeros(a.shape[0], 1, device=x.device)
        self.model(x, t, a[:, None], self.cache, self.start, write=True)
        if self.max_ctx is not None:  # 每层只留最近 max_ctx 个 chunk
            for c in self.cache:
                c["k"], c["v"] = c["k"][-self.max_ctx:], c["v"][-self.max_ctx:]
        self.start += self.fpc
        self.shape = x.shape  # sample_chunk 照这个形状造噪声

    def denoise(self, x, sigma, a):
        # x: [B, C, 5, H, W] 带噪的当前 chunk；sigma: [B]；a: [B] -> 当前 chunk 的速度（只读 cache，不改状态）
        return self.model(x, sigma[:, None] * 1000, a[:, None], self.cache, self.start)

    def sample_chunk(self, a, steps=50, shift=3.0):
        B, C, _, H, W = self.shape  # 只改这一行，原来是 self.ctx[-1].shape
        sigmas = sigma_schedule(steps, shift, a.device)
        x = torch.randn(B, C, self.fpc, H, W, device=a.device)
        for i in range(steps):
            v = self.denoise(x, sigmas[i].expand(B), a)
            x = x + (sigmas[i + 1] - sigmas[i]) * v #x + d(sigma) * d(x)/d(sigma)=target=velocity
        return x

@torch.no_grad()
def rollout(model, context, actions, steps=50, shift=3.0, max_ctx=None, frames_per_chunk=5):
    r = Rollout(model, max_ctx, frames_per_chunk)
    Kc = context.shape[2] // frames_per_chunk #[B, C, T, H, W] -> T // fpc, how many chunk is in context
    for k in range(Kc):  # 先把给定的上下文 commit 进去
        r.commit(context[:, :, k * frames_per_chunk:(k + 1) * frames_per_chunk], actions[:, k])
    video = [context]
    for k in range(Kc, actions.shape[1]):  # per action after context
        x = r.sample_chunk(actions[:, k], steps, shift) #get a new chunk out, already finish denoise steps
        r.commit(x, actions[:, k]) #commit into the context
        video.append(x) #also maintain an output video stream
    return torch.cat(video, dim=2)