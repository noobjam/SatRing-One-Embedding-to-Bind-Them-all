import torch
import torch.nn.functional as F

def seco_loss(z0_1, z0_2, z1_1, z1_2, z2_1, z2_2, z3_1, z3_2, 
               t1, t2, s1, s2, osm1, osm2, temp=0.1, season_thresh=90):
    """
    Compute SeCo loss for 4 subspaces (added OSM semantic).
    Args:
        z0_1, z0_2: Embeddings for Z0 (All Invariant) [B, D]
        z1_1, z1_2: Embeddings for Z1 (Season Variant / Sensor Invariant) [B, D]
        z2_1, z2_2: Embeddings for Z2 (Sensor Variant / Season Invariant) [B, D]
        z3_1, z3_2: Embeddings for Z3 (Semantic - OSM land use) [B, D]
        t1, t2: Timestamps (Day of Year) [B] or [B, T]
        s1, s2: Sensor IDs [B] or [B, T]
        osm1, osm2: OSM land use tags [B] (integer encoded)
        temp: Temperature
        season_thresh: Threshold for same season (days)
    """
    
    # Normalize embeddings
    z0_1 = F.normalize(z0_1, dim=1)
    z0_2 = F.normalize(z0_2, dim=1)
    z1_1 = F.normalize(z1_1, dim=1)
    z1_2 = F.normalize(z1_2, dim=1)
    z2_1 = F.normalize(z2_1, dim=1)
    z2_2 = F.normalize(z2_2, dim=1)
    z3_1 = F.normalize(z3_1, dim=1)
    z3_2 = F.normalize(z3_2, dim=1)
    
    batch_size = z0_1.shape[0]
    device = z0_1.device
    
    # Handle timestamps/sensors if they are sequences
    if t1.dim() > 1: t1 = t1.float().mean(dim=1)
    if t2.dim() > 1: t2 = t2.float().mean(dim=1)
    if s1.dim() > 1: s1 = s1[:, 0]
    if s2.dim() > 1: s2 = s2[:, 0]
    
    # 1. Loss All (Z0)
    l_all = contrastive_loss(z0_1, z0_2, temp)
    
    # 2. Loss Season (Z1)
    time_diff = torch.abs(t1 - t2)
    time_diff = torch.min(time_diff, 365 - time_diff)
    mask_season = (time_diff < season_thresh).float()
    l_season = weighted_contrastive_loss(z1_1, z1_2, mask_season, temp)
    
    # 3. Loss Sensor (Z2)
    mask_sensor = (s1 == s2).float()
    l_sensor = weighted_contrastive_loss(z2_1, z2_2, mask_sensor, temp)
    
    # 4. Loss Semantic (Z3) - NEW
    # Positive if same OSM land use tag
    # osm1/osm2 are integer encoded (0=unknown, 1=forest, 2=agriculture, etc.)
    mask_semantic = (osm1 == osm2).float()
    # Don't penalize if both are unknown (0)
    unknown_mask = ((osm1 == 0) | (osm2 == 0)).float()
    mask_semantic = mask_semantic * (1 - unknown_mask)
    
    l_semantic = weighted_contrastive_loss(z3_1, z3_2, mask_semantic, temp)
    
    return l_all, l_season, l_sensor, l_semantic

def contrastive_loss(z_a, z_b, temp):
    # Standard SimCLR loss
    # z_a, z_b: [B, D]
    # Returns scalar loss
    
    # Cosine similarity
    # logits: [B, B]
    logits = torch.matmul(z_a, z_b.T) / temp
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    
    # We also need to consider z_a vs z_a (excluding self) as negatives?
    # Simplified version: z_a vs z_b
    
    loss = F.cross_entropy(logits, labels)
    return loss

def weighted_contrastive_loss(z_a, z_b, mask, temp):
    # z_a, z_b: [B, D]
    # mask: [B] - 1 if (a[i], b[i]) is a valid positive pair, 0 otherwise
    
    logits = torch.matmul(z_a, z_b.T) / temp
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    
    # Cross entropy
    # We only compute loss for indices where mask == 1
    # But we still use the full batch for negatives (denominator of softmax)
    
    loss_per_sample = F.cross_entropy(logits, labels, reduction='none')
    
    # Apply mask
    masked_loss = loss_per_sample * mask
    
    # Average over valid positives
    if mask.sum() > 0:
        return masked_loss.sum() / mask.sum()
    else:
        return torch.tensor(0.0, device=z_a.device, requires_grad=True)
