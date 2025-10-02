import timm
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from encoder_base import EncoderBase
from torchgeo.models import (
    DOFALarge16_Weights,
    ResNet50_Weights,
    dofa_large_patch16_224,
    resnet50,
)


class S2Encoder(EncoderBase, nn.Module):
    def __init__(self, in_channels: int = 13, backbone="resnet50"):
        super().__init__(in_channels=in_channels)
        self.in_channels = in_channels
        self.backbone = backbone
        self.encoder = self._create_encoder()

        # Sentinel-2 L1C wavelengths (in micrometers)

        self.wavelengths = [
            0.443,
            0.490,
            0.560,
            0.665,
            0.705,
            0.740,
            0.783,
            0.842,
            0.865,
            0.945,
            1.375,
            1.610,
            2.190,
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
