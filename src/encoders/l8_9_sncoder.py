import torch
import torch.nn as nn
from encoder_base import EncoderBase
from torchgeo.models import (
    DOFALarge16_Weights,
    ResNet50_Weights,
    dofa_large_patch16_224,
    resnet50,
)


class EncoderL8(EncoderBase, nn.Module):
    def __init__(self, in_channels: int = 7, backbone="resnet50"):
        super().__init__(in_channels=in_channels)
        self.in_channels = in_channels
        self.backbone = backbone
        self.encoder = self._create_encoder()

        # Landsat 8/9 L2SP wavelengths - 7 optical bands (in micrometers)
        self.wavelengths = [
            0.44,
            0.48,
            0.56,
            0.655,
            0.865,
            1.61,
            2.20,
        ]

    def _create_encoder(self):
        weights = DOFALarge16_Weights.DOFA_MAE
        model = dofa_large_patch16_224(weights=weights)

        # Remove only the final classification head
        if hasattr(model, "head"):
            model.head = nn.Identity()
        elif hasattr(model, "fc"):
            model.fc = nn.Identity()

        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, self.wavelengths)  # [B, 1024]
        return x
