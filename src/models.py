import torch 
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads = 8, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
    def forward(self, x):
        B,T,N,D = x.shape  # Batch, Time, Num_tokens, Dim
        x_out = []
        for i in range(N):
            token_seq = x[:,:,i,:]  # (B,T,D)
            token_seq_norm = self.norm(token_seq)
            attn_out, _ = self.attn(token_seq_norm, token_seq_norm, token_seq_norm)
            token_seq = token_seq + attn_out
            token_seq = token_seq + self.mlp(self.norm(token_seq))
            x_out.append(token_seq.unsqueeze(2))  # (B,T,1,D)
        x_out = torch.cat(x_out, dim=2)  # (B,T,N
        return x_out
    


class TemporalSummerizer(nn.Module):
    def __init__(self, dim,num_heads=8, num_summary_tokens = 196):
        super().__init__()
        self.summary_tokens = nn.Parameter(torch.randn(1,num_summary_tokens,dim))
        self.attn = nn.MultiheadAttention(embed_dim=dim,num_heads=num_heads,batch_first=True)
        self.norm = nn.LayerNorm(dim)


    def forward(self,x):
        B,T,N_minus_1,D = x.shape
        kv = x.view(B, T*N_minus_1,D)
        q = self.summary_tokens.expand(B,-1,-1)
        summary_feat,_ = self.attn(q,kv,kv)
        summary_feat = self.norm(summary_feat)
        return summary_feat


class ConvRefinedBlock(nn.Module):
    def __init__(self, dim, H=14,W=14):
        super().__init__()
        self.H = H
        self.W = W
        self.conv= nn.Sequential(
            nn.Conv2d(dim, dim,kernel_size=3,padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim,kernel_size=3,padding=1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B,T,N,D = x.shape
        x = x[:,:,1:,:]  # remove cls token (B,T,N-1,D)
        x = x.view(B*T, self.H, self.W, D).permute(0,3,1,2)  # (B*T,D,H,W)
        x = self.conv(x)  # (B*T,D,H,W)
        x = x.permute(0,2,3,1).view(B,T,N-1,D)  # (B,T,N-1,D)
        return x
    

class SinTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps):
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(0)  #[1,T]

        B,T = timesteps.shape
        device = timesteps.device
        half= self.dim // 2
        freqs = torch.exp(- torch.arange(half, device=device) * (torch.log(torch.tensor(10000.0)) / half))
        args = timesteps.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # [B,T,half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B,T,dim]
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1), mode='constant')
        return emb  # [B,T,dim]
    

class ConvDecoder(nn.Module):
    def __init__(self, in_dim=1024, out_dim=64, H=14, W=14, upsample_factor=16):
        super().__init__()
        self.H = H
        self.W = W
        self.upsample_factor = upsample_factor
        self.conv = nn.Sequential(
        nn.Conv2d(in_dim, in_dim//2, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(in_dim//2, in_dim//4, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(in_dim//4, out_dim, 3, padding=1)
        )


    def forward(self, x):
        B, N, D = x.shape
        H = W = int(N**0.5)
        x = x.view(B, H, W, D).permute(0,3,1,2).contiguous()
        x = F.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x
    


class TRACE(nn.Module):
    def __init__(self, dim=1024, num_heads=8, temporal_depth=2, H=14, W=14, aggregate='summarizer', max_timesteps=90, learned_emb=False):
        super().__init__()
        self.temporal_blocks = nn.ModuleList([TemporalAttentionBlock(dim=dim, num_heads=num_heads) for _ in range(temporal_depth)])
        self.conv_refine = ConvRefinedBlock(dim=dim, H=H, W=W)
        self.norm = nn.LayerNorm(dim)
        self.aggregate = aggregate
        self.learned_time_emb = learned_emb
        if learned_emb: self.time_emb = nn.Embedding(max_timesteps, dim)
        else: self.time_emb = SinTimeEmbeddings(dim)

        if self.aggregate == 'summarizer':
            num_summary_tokens = H * W
            self.summarizer = TemporalSummerizer(dim=dim, num_heads=num_heads, num_summary_tokens=num_summary_tokens)

        self.decoder = ConvDecoder(in_dim=dim, out_dim=64, H=H, W=W, upsample_factor=16)

    def forward(self, x, timesteps=None, return_type='summary_pixel'):
        B, T, N, D = x.shape
        if timesteps is None:
            timesteps = torch.arange(T, device=x.device).unsqueeze(0).repeat(B, 1)
        t_emb = self.time_emb(timesteps)
        x = x + t_emb.unsqueeze(2)
        for block in self.temporal_blocks: x = block(x)
        x_conv = self.conv_refine(x)
        x_processed = x[:, :, 1:, :] + x_conv
        x_processed = self.norm(x_processed) # Shape: [B, T, N-1, D]

        # --- BRANCHING LOGIC FOR INFERENCE ---
        if return_type == 'summary_pixel' or return_type == 'summary_global':
            # This path produces both summary types
            if self.aggregate == 'mean': x_feat = x_processed.mean(dim=1)
            elif self.aggregate == 'last': x_feat = x_processed[:, -1, :, :]
            elif self.aggregate == 'summarizer': x_feat = self.summarizer(x_processed)
            else: raise ValueError(f"Unknown aggregation type: {self.aggregate}")
            
            if return_type == 'summary_global':
                return x_feat.mean(dim=1) # [B, D]
            else: # summary_pixel
                return self.decoder(x_feat) # [B, C, H, W]

        elif return_type == 'per_frame_pixel':
            x_per_frame = x_processed.view(B * T, N - 1, D)
            pixel_emb_per_frame = self.decoder(x_per_frame)
            _, C, H, W = pixel_emb_per_frame.shape
            return pixel_emb_per_frame.view(B, T, C, H, W)

        else:
            raise ValueError(f"Unknown return_type: {return_type}")

        

# B, T, N, D = 10, 5, 197, 1024
# x = torch.randn(B, T, N, D)
# timesteps = torch.arange(T).unsqueeze(0).expand(B, -1)
# model = TRACE(dim=1024, num_heads=8, temporal_depth=2, learned_time_emb=False)
# out = model(x, timesteps) # [B, N, D]



