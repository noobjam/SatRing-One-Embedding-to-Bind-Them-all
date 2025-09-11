import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class EncoderBase(nn.Module, ABC):
    """
    Base class for modality encoders that output an embedding vector in a shared latent space.
    Subclasses must implement forward(x) and return a tensor of shape (B, embed_dim).

    This base also provides an encode() helper that optionally L2-normalizes embeddings,
    which is commonly required for contrastive losses (InfoNCE).
    """
    def __init__(self, embed_dim: int = 256, normalize: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.normalize = normalize

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, embed_dim)
        """
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs forward and applies optional L2 normalization.
        """
        z = self.forward(x)
        if self.normalize:
            z = nn.functional.normalize(z, dim=-1)
        return z
