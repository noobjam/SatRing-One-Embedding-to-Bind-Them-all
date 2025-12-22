import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, T, N, D = x.shape
        x_flat = x.view(B * T, N, D)
        attn_out, _ = self.attn(self.norm1(x_flat), x_flat, x_flat)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.view(B, T, N, D)


class TemporalSummarizer(nn.Module):
    def __init__(self, dim, num_heads=8, num_tokens=196):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, N, D = x.shape
        kv = x.view(B, T * N, D)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, kv, kv)
        return self.norm(out)  # [B, 196, D]


class ConvRefineBlock(nn.Module):
    def __init__(self, dim, H=14, W=14):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.H, self.W = H, W

    def forward(self, x):
        B, T, N, D = x.shape
        x = x[:, :, 1:, :]  # remove CLS
        x = x.reshape(B * T, self.H, self.W, D).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).reshape(B, T, 196, D)
        return x


class SinTimeEmbedding(nn.Module):
    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(0)
        B, T = t.shape
        device, half_dim = t.device, self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=device)
            * (torch.log(torch.tensor(10000.0)) / half_dim)
        )
        args = t.unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def __init__(self, dim):
        super().__init__()
        self.dim = dim


class ConvDecoder(nn.Module):
    def __init__(self, in_dim=1024, out_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
        )

    def forward(self, x):
        B, N, D = x.shape
        x = x.view(B, 14, 14, D).permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=16, mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = self.refine(x)
        return x  # [B, out_dim, 224, 224]


class TRACE(nn.Module):
    def __init__(self, dim=1024, out_dim=256, num_heads=8, depth=4):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim

        self.time_emb = SinTimeEmbedding(dim)
        self.temporal_blocks = nn.ModuleList(
            [TemporalAttentionBlock(dim, num_heads) for _ in range(depth)]
        )
        self.refine = ConvRefineBlock(dim)
        self.norm = nn.LayerNorm(dim)
        self.summarizer = TemporalSummarizer(dim, num_heads)
        self.decoder = ConvDecoder(dim, out_dim)

        # Four SeCo subspaces (added OSM semantic)
        make_head = lambda: nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 128),
            nn.BatchNorm1d(128),
        )
        self.head_all = make_head()  # Z0: invariant to everything
        self.head_season = make_head()  # Z1: seasonal invariant
        self.head_sensor = make_head()  # Z2: sensor invariant
        self.head_semantic = make_head()  # Z3: semantic invariant (OSM)

    def encode(self, x, timesteps):
        B, T, N, D = x.shape
        x = x + self.time_emb(timesteps)[:, :, None, :]
        for blk in self.temporal_blocks:
            x = blk(x)
        x = x[:, :, 1:, :] + self.refine(x)
        return self.norm(x)  # [B, T, 196, D]

    def forward(self, x, timesteps=None, flavor="sequence"):
        """
        Forward pass with three inference modes.
        
        Args:
            x: [B, T, N, D] - Encoded tokens
            timesteps: [B, T] - Day of year
            flavor: 'global', 'sequence', or 'snapshot'
        
        Returns:
            [B, out_dim, 224, 224] - Spatial embeddings at 10m resolution
        """
        B, T = x.shape[0], x.shape[1]
        
        if timesteps is None:
            timesteps = torch.arange(T, device=x.device)[None].repeat(B, 1)

        tokens = self.encode(x, timesteps)  # [B, T, 196, D]

        if flavor == "global":
            # Aggregate all timesteps into single representation
            tokens = self.summarizer(tokens)  # [B, 196, D]
            
        elif flavor == "sequence":
            # Process each timestep, then average
            B, T, N, D = tokens.shape
            decoded = self.decoder(tokens.view(B * T, N, D))  # [B*T, out_dim, 224, 224]
            decoded = decoded.view(B, T, self.out_dim, 224, 224)
            # Average over time
            return decoded.mean(dim=1)  # [B, out_dim, 224, 224]
            
        elif flavor == "snapshot":
            # Select single timestep
            if self.training:
                idx = torch.randint(0, T, (B,), device=x.device)
            else:
                idx = torch.full((B,), T // 2, dtype=torch.long, device=x.device)
            tokens = tokens[torch.arange(B, device=x.device), idx]  # [B, 196, D]

        return self.decoder(tokens)  # [B, out_dim, 224, 224]

    def project(self, emb_10m):
        x = F.adaptive_avg_pool2d(emb_10m, 1).flatten(1)
        return (
            F.normalize(self.head_all(x), dim=1),
            F.normalize(self.head_season(x), dim=1),
            F.normalize(self.head_sensor(x), dim=1),
        )


# Test
if __name__ == "__main__":
    x = torch.randn(2, 25, 197, 1024)
    t = torch.randint(0, 365, (2, 25))
    model = TRACE()

    print(model(x, t, "global").shape)  # [2, 256, 224, 224]
    print(model(x, t, "sequence").shape)  # [2, 256, 224, 224]
    print(model(x, t, "snapshot").shape)  # [2, 256, 224, 224]
