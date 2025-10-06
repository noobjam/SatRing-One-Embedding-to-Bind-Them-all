from abc import ABC, abstractmethod

import torch
import torch.nn as nn
# import torchvision.models as models


class EncoderBase(ABC):
    def __init__(self, in_channels: int, backbone: str = "resnet50"):
        super().__init__()
        self.in_channels = in_channels
        self.backbone = backbone
        self.encoder = self._create_encoder()

    @abstractmethod
    def _create_encoder(self):
        pass
