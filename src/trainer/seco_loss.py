import torch
import torch.nn.functional as F


def satring_loss(z_snap1, z_snap2, z_seq1, z_seq2, z_year1, z_year2, temp=0.1):
    """
    Compute 3-Flavor SSL loss for SatRing.
    Args:
        z_snap1, z_snap2: Snapshot embeddings [B, D] - temporal variant
        z_seq1, z_seq2: Sequence embeddings [B, D] - trajectory shape
        z_year1, z_year2: Year embeddings [B, D] - temporally invariant
        temp: Temperature
    """

    # 1. Snapshot Loss (Contrastive)
    # Learns per-date features robust to spatial augmentations
    l_snap = contrastive_loss(z_snap1, z_snap2, temp)

    # 2. Sequence Loss (Contrastive)
    # Learns trajectory shape robust to spatial augmentations
    l_seq = contrastive_loss(z_seq1, z_seq2, temp)

    # 3. Year Loss (Contrastive)
    # Learns temporally invariant features (same location, different date samples)
    l_year = contrastive_loss(z_year1, z_year2, temp)

    return l_snap, l_seq, l_year


def contrastive_loss(z_a, z_b, temp=0.1):
    """
    Standard NT-Xent (SimCLR) style contrastive loss.
    """
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)

    # logits: [B, B]
    logits = torch.matmul(z_a, z_b.T) / temp
    labels = torch.arange(z_a.shape[0], device=z_a.device)

    loss = F.cross_entropy(logits, labels)
    return loss


def sequence_reconstruction_loss(pred_tokens, target_tokens, mask):
    """
    Masked Temporal Reconstruction Loss (Placeholder for future Phase 2/3).
    Args:
        pred_tokens: [B, T, D] predicted tokens
        target_tokens: [B, T, D] ground truth tokens from frozen DOFA
        mask: [B, T] binary mask of which steps were masked
    """
    # MSE loss on masked positions
    mse = F.mse_loss(pred_tokens, target_tokens, reduction="none")
    # Average only over masked positions
    loss = (mse.mean(dim=-1) * mask).sum() / (mask.sum() + 1e-6)
    return loss
