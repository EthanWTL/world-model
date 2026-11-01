import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),  # 位置 m = 0, 1, ..., 1023
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))  # 每一对的速度 θ_k
    # freqs[m, k] = m * θ_k：位置 m 的第 k 对要转多少度
    return torch.polar(torch.ones_like(freqs), freqs) #[r=1, freqs] -> [m*1 + n*i]

def rope_apply(x, grid_size, freqs, start=0):
    #x = [2, 7680, 16, 72]
    b, L, n, d = x.shape #[2, 7680, 16, 72]
    c = d // 2  #c = 36
    f, h, w = grid_size # [40, 12,16]

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1) #(3*[1024,12])

    freqs_i = torch.cat([
        freqs[0][start:start + f].view(f, 1, 1, -1).expand(f, h, w, -1), #[40, 12] -> [40,1,1,12] -> [40,12,16,12]
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1), #[12,12] -> [1,12,1,12] -> [40,12,16,12]
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1), #[16,12] -> [1,1,16,12] -> [40,12,16,12]
    ], dim=-1).reshape(L, 1, -1) #[40,12,16,36] -> [7680,1,36]

    #x = [2, 7680, 16, 72] -> [2,7680, 16, 36,2] -> in complex, x_c = [2,7680, 16, 36]
    x_c = torch.view_as_complex(x.to(torch.float64).reshape(b, L, n, -1, 2)) 

    # [2,7680, 16, 36] * [7680,1,36] -> [2, 7680, 16, 36] -> in real, [2, 7680, 16, 72]
    return torch.view_as_real(x_c * freqs_i).flatten(3).type_as(x) 

def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2  # 128 种转速
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))  # [N, 128]：位置 × 转速 = 角度
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)  # [N, 256]

def chunk_mask(num_chunks, tokens_per_chunk, window, device):
    chunk_id = torch.arange(num_chunks * tokens_per_chunk, device=device) // tokens_per_chunk  # [00,,,5555,,,777] [7680]
    q = chunk_id[:, None]  # [L, 1]：谁在看
    k = chunk_id[None, :]  # [1, L]：被看的
    mask = k <= q  # causal：只能看自己和前面的 chunk
    if window is not None:
        mask = mask & (k >= q - window)  # 滑动窗口：最多往前看 window 个 chunk
    return mask  # [L, L]，True = 可以看

def chunk_attention(q, k, v, num_chunks, window=None):
    # q, k, v: [B, n, L, d]。第 i 个 chunk 的 query 只和第 max(0, i - window) ... i 个 chunk 的 key/value 做注意力
    # 和 chunk_mask 完全等价，但每次调用都不需要 mask：被挡住的块根本不算，还能用上 FlashAttention
    T = q.shape[2] // num_chunks  # 每个 chunk 的 token 数（960）
    out = []
    for i in range(num_chunks):
        s = 0 if window is None else max(0, i - window)  #能看的第一个chunk 如果没有window s=i
        #every chunk do their own self attention
        out.append(F.scaled_dot_product_attention(q[:, :, i * T:(i + 1) * T],
                                                  k[:, :, s * T:(i + 1) * T], v[:, :, s * T:(i + 1) * T]))
    return torch.cat(out, dim=2)  # 8 段输出按 token 维拼回 [B, n, L, d]

class WanModel(nn.Module):
    def __init__(
        self, 
        patch_size=(1, 2, 2), 
        in_dim=16, 
        out_dim = 16, 
        dim=1152, 
        num_heads=16, 
        ffn_dim = 4608, 
        num_layers = 12,
        freq_dim = 256, 
        num_actions = 13,
        action_dim = 72,
        causal = False,
        window = None
        ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size = patch_size, stride = patch_size)

        self.out_dim = out_dim
        self.head = Head(dim, out_dim, patch_size)

        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)])

        d = dim // num_heads # d = 72
        self.freqs = torch.cat([
            rope_params(1024, d - (d // 6) * 4),
            rope_params(1024, (d // 6) * 2),
            rope_params(1024, (d // 6) * 2)
        ], dim=1) #[1024, 36]
        
        self.dim = dim
        self.freq_dim = freq_dim
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.action_embedding = nn.Embedding(num_actions, action_dim)
        self.action_proj = nn.Linear(action_dim, dim)

        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.causal = causal
        self.window = window

        self.init_weights()
        
    def forward(self, x, t, a, cache=None, start=0, write=False):
        #move rope table into device
        if self.freqs.device != x.device:
            self.freqs = self.freqs.to(x.device)

        #x = [2, 16, 40, 24, 32] [B, C, T, H, W]
        x = self.patch_embedding(x) # x = [2, 1152, 40, 12, 16] [B, C', T, H/2, W/2]
        grid_size = tuple(x.shape[2:]) #grid_size = [40, 12, 16] [T, H/2, W/2]
        x = x.flatten(2).transpose(1,2) # x = [2, 7680, 1152] [B, N, D]

        #time embedding
        B,L = x.shape[0], x.shape[1]
        K = t.shape[1]
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).float()) #[16, 1152] [B*K, dim]
        e = e.view(B,K,-1) #[2,8,1152] [B, K, dim]
        e = e + self.action_proj(self.action_embedding(a)) #[2,8,1152] [B, K, dim]
        e = e.repeat_interleave(L // K, dim=1)  # [7680 // 8] [2,7680,1152] share one timestep every 960 tokens (1 chunk)
        e0 = self.time_projection(e).unflatten(2, (6, self.dim)) #[2,7680, 6912] -> [2, 7680, 6, 1152]


        #causal mask
        #mask = chunk_mask(K, L // K, self.window, x.device) if self.causal else None
        #or causal chunk attention
        causal = (K, self.window) if self.causal else None

        #self attention
        for i, block in enumerate(self.blocks):
            x = block(x, e0, grid_size, self.freqs, causal, None if cache is None else cache[i], start, write)

        #split one token in 1152 back to multiple tokens with features length 16.
        x = self.head(x, e) # [2,7680, 64] # down dimention from 1152 into 64
        x = self.unpatchify(x, grid_size)
        return x #[2, 16, 40, 24, 32]

    def unpatchify(self, x, grid_size):
        B = x.shape[0]
        f, h, w = grid_size
        pt, ph, pw = self.patch_size
        c = self.out_dim
        x = x.view(B, f, h, w, pt, ph, pw, c) #x = [2, 40, 12, 16, 1, 2, 2, 16]
        x = torch.einsum('bfhwpqrc->bcfphqwr', x)# x = [2, 16, 40, 1, 12, 2, 16, 2]
        return x.reshape(B, c, f * pt, h * ph, w * pw) # x = [2, 16, 40 * 1, 12 * 2, 16 * 2]

    def init_weights(self):
        # 所有 Linear：xavier，bias 置 0（self.modules() 会递归遍历每一层里的每个子模块）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # patch embedding 当成 64 -> 1152 的 Linear 来初始化
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        # 时间嵌入用小一点的初始化
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        # 出口置 0：一开始模型输出全是 0
        nn.init.zeros_(self.head.head.weight)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, eps = 1e-6):
        super().__init__()
        self.num_heads = num_heads # 16
        self.head_dim = dim // num_heads # 72
        self.q = nn.Linear(dim, dim) #(1152, 1152)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim) #communicate between heads within one token

        self.norm_q = WanRMSNorm(dim, eps)
        self.norm_k = WanRMSNorm(dim, eps)

    def forward(self, x, grid_size, freqs, causal, cache = None, start=0, write=False):
        b, L, _ = x.shape #[2,7680, _1152]
        n, d = self.num_heads, self.head_dim #(16,72)
        q = self.norm_q(self.q(x)).view(b, L, n, d) #[2,7680, 1152].view(2, 7680, 16,72) #split the 1152 into 16 heads each with 72 features
        k = self.norm_k(self.k(x)).view(b, L, n, d)
        v = self.v(x).view(b, L, n, d)

        #3d rotation
        q = rope_apply(q, grid_size, freqs, start)
        k = rope_apply(k, grid_size, freqs, start)

        #SPDA need [B, n, L, d] [2, 16, 7680, 72]
        #out = F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2), attn_mask = mask) #[2, 16, 7680, 72]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if cache is not None:
            ks, vs = cache['k'] + [k], cache['v'] + [v]
            if write:
                cache['k'], cache['v'] = ks, vs #update cache
            out = F.scaled_dot_product_attention(q, torch.cat(ks, dim=2), torch.cat(vs, dim=2))
        elif causal is None:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = chunk_attention(q,k,v,*causal)
        out = out.transpose(1,2).flatten(2)#[2, 7680, 16, 72]. flatten(2) -> [2,7680, 1152]
        return self.o(out)


class WanAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = WanSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), #(1152, 4608)
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim) #(4608, 1152)
        )
        self.modulation = nn.Parameter(torch.randn(1,6,dim) / dim ** 0.5) #block level changes for the embedding [1, 6, 1152]

    def forward(self, x , e, grid_size, freqs, causal = None, cache=None, start = 0, write = False):
        # [1, 6,1152] -> [1,1,6,1152] + [2, 7680, 6, 1152] -> [2, 7680, 6, 1152] -> (6*[2, 7680, 1165])
        shift1, scale1, gate1, shift2, scale2, gate2 = (self.modulation.unsqueeze(0) + e).unbind(2) 
        #x [2 7680, 1152] scale1[2,7680,1152]
        x = x + self.self_attn(self.norm1(x) * (1 + scale1) + shift1, grid_size, freqs, causal,cache, start, write) * gate1 #[2, 7680, 1152] (qkv heads, o)
        x = x + self.ffn(self.norm2(x) * (1+scale2) + shift2) * gate2 #[2, 7680, 1152] (up, gelu, down)
        return x


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_f = x.float()
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)  # 平方 mean +e 再开方 再除
        return x_f.type_as(x) * self.weight #给一个update weight

class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        # e: [B, L, D] -> [B, L, 1, D]，加上 [1, 1, 2, D] -> [B, L, 2, D]
        shift, scale = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).unbind(2)
        return self.head(self.norm(x) * (1 + scale) + shift)