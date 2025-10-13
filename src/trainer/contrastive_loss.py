
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



def info_nce_loss(z_a,z_b,temperature=0.5):
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)
    batch_size = z_a.shape[0]
    logits = torch.matmul(z_a, z_b.T) / temperature
    labels = torch.arange(batch_size).to(z_a.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    loss = (loss_a + loss_b) / 2
    return loss


