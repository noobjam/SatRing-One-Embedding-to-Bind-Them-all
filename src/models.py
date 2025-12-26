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
        # Self-attention within each patch across time is handled by TRACE's structure
        # Here we do spatial attention within a time-step if N tokens are passed
        attn_out, _ = self.attn(self.norm1(x_flat), x_flat, x_flat)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.view(B, T, N, D)


class SinTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(0)
        B, T = t.shape
        device, half_dim = t.device, self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, device=device)
            * (torch.log(torch.tensor(10000.0)) / half_dim)
        )
        args = t.unsqueeze(-1) * freqs.to(t.dtype)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class ProgressiveDecoder(nn.Module):
    """
    Progressive spatial upsampler (U-Net/FPN style) to 10m resolution.
    [B, D, 14, 14] -> [B, out_dim, 224, 224]
    """

    def __init__(self, in_dim=1024, out_dim=256):
        super().__init__()

        # Stage 1: 14 -> 28
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, in_dim // 2),
            nn.GELU(),
        )

        # Stage 2: 28 -> 56
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim // 2, in_dim // 4, kernel_size=4, stride=2, padding=1
            ),
            nn.GroupNorm(8, in_dim // 4),
            nn.GELU(),
        )

        # Stage 3: 56 -> 112
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim // 4, in_dim // 8, kernel_size=4, stride=2, padding=1
            ),
            nn.GroupNorm(8, in_dim // 8),
            nn.GELU(),
        )

        # Stage 4: 112 -> 224
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim // 8, out_dim, kernel_size=4, stride=2, padding=1
            ),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )

        # Final refinement
        self.final = nn.Conv2d(out_dim, out_dim, 3, padding=1)

    def forward(self, x):
        # x: [B, 196, D] or [B, D, 14, 14]
        if x.dim() == 3:
            B, N, D = x.shape
            H = W = int(N**0.5)
            x = x.view(B, H, W, D).permute(0, 3, 1, 2)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.final(x)


class TRACE(nn.Module):
    def __init__(self, dim=1024, out_dim=256, num_heads=8, depth=4):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim

        self.time_emb = SinTimeEmbedding(dim)

        self.temporal_blocks = nn.ModuleList(
            [TemporalAttentionBlock(dim, num_heads) for _ in range(depth)]
        )

        # Temporal aggregator for Sequence flavor
        self.temporal_agg = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.agg_norm = nn.LayerNorm(dim)

        self.norm = nn.LayerNorm(dim)
        self.decoder = ProgressiveDecoder(dim, out_dim)

        # SSL Projection Heads (Lightweight)
        make_head = lambda: nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, 128),
        )
        self.proj_snapshot = make_head()
        self.proj_sequence = make_head()
        self.proj_year = make_head()

    def encode(self, x, timesteps):
        """
        Input Enrichment & Temporal Blocks
        x: [B, T, N, D]
        """
        B, T, N, D = x.shape

        # 1. Temporal Enrichment
        x = x + self.time_emb(timesteps)[:, :, None, :]

        # 2. Main Temporal Blocks
        for blk in self.temporal_blocks:
            x = blk(x)

        return x  # [B, T, N, D]

    def forward(self, x, timesteps, flavor="sequence"):
        """
        3-Flavor Routing
        """
        B, T, N, D = x.shape

        # Encode all tokens
        tokens = self.encode(x, timesteps)  # [B, T, N, D]

        # We typically drop the CLS token [0] if coming from DOFA for spatial reconstruction
        # Assuming DOFA tokens are [CLS, P1, P2, ..., P196]
        spatial_tokens = tokens[:, :, 1:, :]  # [B, T, 196, D]

        if flavor == "snapshot":
            # Select random or middle timestep
            if self.training:
                idx = torch.randint(0, T, (B,), device=x.device)
            else:
                idx = torch.full((B,), T // 2, dtype=torch.long, device=x.device)
            latent = spatial_tokens[
                torch.arange(B, device=x.device), idx
            ]  # [B, 196, D]

        elif flavor == "sequence":
            # Aggregate over time for each spatial token
            # tokens: [B, T, 196, D] -> [B, 196, T, D]
            tokens_pixel = spatial_tokens.permute(0, 2, 1, 3).reshape(B * 196, T, D)
            # Use self-attention or mean as a simpler baseline
            # Here we use mean for 'sequence' to keep it robust
            latent = tokens_pixel.mean(dim=1).view(B, 196, D)

        elif flavor == "year":
            # Year is also a spatial map but invariant to season
            latent = spatial_tokens.mean(dim=1)  # [B, 196, D]

        else:
            raise ValueError(f"Unknown flavor: {flavor}")

        # Map to 10m embedding field
        return self.decoder(latent)

    def project(self, emb_10m, flavor="snapshot"):
        # Pool across spatial dimensions to get scalar representation for SSL
        h = F.adaptive_avg_pool2d(emb_10m, 1).flatten(1)
        if flavor == "snapshot":
            return F.normalize(self.proj_snapshot(h), dim=1)
        elif flavor == "sequence":
            return F.normalize(self.proj_sequence(h), dim=1)
        elif flavor == "year":
            return F.normalize(self.proj_year(h), dim=1)
        return h


# Test
if __name__ == "__main__":
    x = torch.randn(2, 5, 197, 1024)
    t = torch.randint(0, 365, (2, 5))
    c = torch.randn(2, 2)
    model = TRACE()

    for f in ["snapshot", "sequence", "year"]:
        out = model(x, t, coords=c, flavor=f)
        print(f"Flavor {f}: {out.shape}")  # [2, 256, 224, 224]
