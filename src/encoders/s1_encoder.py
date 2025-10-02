# from .encoder_base import EncoderBase
# import torch.nn as nn


# class S1_Encoder(EncoderBase):
#     """
#     Sentinel-1 encoder (e.g., GRD VV/VH), expecting input shape (B, 2, H, W).
#     Uses a CNN backbone (default: torchvision resnet18) adapted to 2 channels and outputs embed_dim features.
#     """
#     def __init__(self, embed_dim: int = 256, backbone_type: str = 'resnet18', **backbone_kwargs):
#         super().__init__(embed_dim)
#         self.in_channels = 2  # S1 VV, VH

#         if 'resnet' in backbone_type.lower():
#             import torchvision.models as models
#             backbone = getattr(models, backbone_type)(weights=None, **backbone_kwargs)
#             backbone.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#             backbone.fc = nn.Linear(backbone.fc.in_features, embed_dim)
#             self.encoder = backbone
#         else:
#             import timm
#             self.encoder = timm.create_model(backbone_type, pretrained=False, in_chans=self.in_channels,
#                                              num_classes=embed_dim, **backbone_kwargs)

#     def forward(self, x):
#         return self.encoder(x)
