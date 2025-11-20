import timm
import torch
import torch.nn as nn
from encoder_base import EncoderBase
from torchgeo.models import (
    DOFALarge16_Weights,
    dofa_large_patch16_224,
)


class S2EncoderRaw(EncoderBase, nn.Module):
    def __init__(self, in_channels: int = 13):
        super().__init__(in_channels=in_channels)
        self.in_channels = in_channels
        self.encoder = self._create_encoder()
        self.wavelengths = torch.tensor([
            0.443,  # B1
            0.490,  # B2
            0.560,  # B3
            0.665,  # B4
            0.705,  # B5
            0.740,  # B6
            0.783,  # B7
            0.842,  # B8
            0.865,  # B8A
            0.945,  # B9
            1.375,  # B10
            1.610,  # B11
            2.190,  # B12
        ], dtype=torch.float32)

    def _create_encoder(self):

        weights = DOFALarge16_Weights.DOFA_MAE
        model = dofa_large_patch16_224(weights=weights)


        model.head = nn.Identity()

        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wavelengths = torch.tensor(self.wavelengths, dtype=torch.float32, device=x.device)
        
        # 1. Patch embedding with wavelengths (returns tuple)
        x, _ = self.encoder.patch_embed(x, wavelengths)  # Unpack the tuple
        
        # 2. Add CLS token
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # 3. Add position embeddings
        x = x + self.encoder.pos_embed
        
        # 4. Pass through transformer blocks
        for blk in self.encoder.blocks:
            x = blk(x)
        
        # 5. Final layer norm
        x = self.encoder.fc_norm(x)
        
        return x