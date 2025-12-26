import torch
import torch.nn as nn

from encoders.s2_encoder import S2Encoder
from models import TRACE
from wavelengths import SENSOR_WAVELENGTHS


class SeCoModel(nn.Module):
    def __init__(
        self,
        image_encoder_arch="dofa_large",
        trace_dim=1024,
        trace_depth=4,
        trace_heads=8,
        out_dim=256,
        in_channels=7,  # Default to 7 common bands
    ):
        super().__init__()

        # Image Encoder (Spatial)
        # We use S2Encoder as the base class, but we will dynamically pass wavelengths.
        # DOFA is flexible.
        self.image_encoder = S2Encoder(in_channels=in_channels)

        # Temporal Backbone
        self.backbone = TRACE(
            dim=trace_dim, out_dim=out_dim, num_heads=trace_heads, depth=trace_depth
        )

    def forward(
        self, x, timesteps, sensor_ids=None, flavor="sequence", return_projections=True
    ):
        """
        Forward pass with 3-Flavor routing.

        Args:
            x: [B, T, C, H, W] - Batch of image sequences
            timesteps: [B, T] - Day of Year
            sensor_ids: [B, T] - Sensor ID (0=S2, 1=LS)
            flavor: 'snapshot', 'sequence', or 'year'
            return_projections: If True, returns projections for SSL.
        """
        B, T, C, H, W = x.shape

        # Flatten for spatial encoder
        x_flat = x.view(B * T, C, H, W)
        if sensor_ids is None:
            sensor_ids_flat = torch.zeros(B * T, dtype=torch.long, device=x.device)
        else:
            sensor_ids_flat = sensor_ids.view(-1)

        unique_sensors = torch.unique(sensor_ids_flat)
        tokens_list = []
        indices_list = []

        for s_id in unique_sensors:
            mask = sensor_ids_flat == s_id
            if not mask.any():
                continue
            indices = torch.nonzero(mask).squeeze(1)
            x_sub = x_flat[indices]
            wavs = SENSOR_WAVELENGTHS[s_id.item()].to(x.device)

            # Forward through DOFA blocks
            enc = self.image_encoder.encoder
            x_emb, _ = enc.patch_embed(x_sub, wavs)
            cls_token = enc.cls_token.expand(x_sub.shape[0], -1, -1)
            x_emb = torch.cat((cls_token, x_emb), dim=1)
            x_emb = x_emb + enc.pos_embed
            for blk in enc.blocks:
                x_emb = blk(x_emb)
            x_out = enc.fc_norm(x_emb)

            tokens_list.append(x_out)
            indices_list.append(indices)

        # Reassemble tokens: [B, T, N, D]
        N, D = tokens_list[0].shape[1], tokens_list[0].shape[2]
        all_tokens = torch.zeros(
            B * T, N, D, device=x.device, dtype=tokens_list[0].dtype
        )
        for tokens, indices in zip(tokens_list, indices_list):
            all_tokens[indices] = tokens
        all_tokens = all_tokens.view(B, T, N, D)

        # Temporal & Flavor Forward
        spatial_emb = self.backbone(all_tokens, timesteps, flavor=flavor)

        if return_projections:
            # Get projection for the requested flavor
            z = self.backbone.project(spatial_emb, flavor=flavor)
            return spatial_emb, z
        else:
            return spatial_emb

    def get_embeddings(self, x, timesteps, sensor_ids=None, flavor="sequence"):
        self.eval()
        with torch.no_grad():
            return self.forward(
                x, timesteps, sensor_ids, flavor=flavor, return_projections=False
            )
