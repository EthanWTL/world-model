import torch

def add_noise(x0, sigma, frames_per_chunk=5):
    #x0 [2, 16, 40,24,32] 
    #noise [2, 16, 40,24,32]
    #sigma [2,8]
    noise = torch.randn_like(x0)                         
    s = sigma.repeat_interleave(frames_per_chunk, dim=1)  # [2,40]
    s = s[:, None, :, None, None] # [2, 1, 40, 1, 1]
    x_t = (1 - s) * x0 + s * noise                       
    target = noise - x0                                  
    return x_t, target

def sample_sigma(batch, num_chunks, shift=3.0, device="cuda"):
    u = torch.rand(batch, num_chunks, device=device)  
    return shift * u / (1 + (shift - 1) * u)     #after draw a noise u, pretend it's 3 time stronger.      

def clean_prefix(sigma, p=0.5):
    # sigma: [B, K]。每个样本以概率 p，把开头随机长度的一段 chunk 设成干净的（σ = 0）
    B, K = sigma.shape #[4,8]
    use = torch.rand(B, device=sigma.device) < p             # [B]：这个样本要不要加干净前缀
    length = torch.randint(1, K, (B,), device=sigma.device)  # [B]：前缀长度 1 ~ K-1
    length = length * use                                    # 不加前缀的样本，长度变成 0
    chunk_id = torch.arange(K, device=sigma.device)          # [K]：0, 1, ..., 7
    is_clean = chunk_id[None, :] < length[:, None]           # [B, K]：前 length 个 chunk 是 True
    return sigma.masked_fill(is_clean, 0.0), is_clean

def diffusion_forcing_loss(model, x0, a, shift=3.0, p_clean=0.5, frames_per_chunk=5):
    B, K = a.shape
    sigma = sample_sigma(B, K, shift, x0.device)          # [B, K]：每个 chunk 一个噪声水平
    sigma, is_clean = clean_prefix(sigma, p_clean)        # 一半的样本，开头一段变成干净的上下文
    x_t, target = add_noise(x0, sigma, frames_per_chunk)  # 加噪，算出要预测的速度
    pred = model(x_t, sigma * 1000, a)                    # 模型预测速度，t = σ × 1000
    err = (pred - target).pow(2).mean(dim=(1, 3, 4))      # [B, F]：每一帧的均方误差（对通道、高、宽求平均）
    err = err.view(B, K, frames_per_chunk).mean(-1)       # [B, K]：每个 chunk 的均方误差
    keep = ~is_clean                                      # 干净的 chunk 不算 loss
    return (err * keep).sum() / keep.sum()                # 只在带噪的 chunk 上求平均

def sigma_schedule(steps=50, shift=3.0, device="cuda"):
    u = torch.linspace(1, 0, steps + 1, device=device)  # u 从 1 均匀走到 0，51 个点
    return shift * u / (1 + (shift - 1) * u) 
