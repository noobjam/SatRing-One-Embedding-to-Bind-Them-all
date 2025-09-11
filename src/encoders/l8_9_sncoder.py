from .encoder_base import EncoderBase
import torch.nn as nn
import torchvision

class L8_9_Encoder(EncoderBase):
    def __init__(self, embed_dim: int = 256, backbone_type: str = 'resnet18', **backbone_kwargs):
        super().__init__(embed_dim)
        self.in_channels = 11  # Landsat 8/9 has 11 bands

        if 'resnet' in backbone_type.lower():
            import torchvision.models as models
            backbone = getattr(models, backbone_type)(weights=None, **backbone_kwargs)
            backbone.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            backbone.fc = nn.Linear(backbone.fc.in_features, embed_dim)
            self.encoder = backbone
        else:
            import timm
            self.encoder = timm.create_model(backbone_type, pretrained=False, in_chans=self.in_channels,
                                             num_classes=embed_dim, **backbone_kwargs)

    def forward(self, x):
        return self.encoder(x)
