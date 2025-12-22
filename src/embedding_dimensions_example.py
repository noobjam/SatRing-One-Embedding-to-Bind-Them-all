"""
Quick reference for embedding dimension control in SeCo model.
"""

# ============================================================================
# EMBEDDING DIMENSION CONTROL
# ============================================================================

# Level 1: Spatial Embedding Dimension (out_dim)
# ------------------------------------------------
# Controls: [B, out_dim, 224, 224] - The final 10m resolution embeddings
# Where: SeCoModel initialization

from seco_model import SeCoModel

# Option A: Small (128 channels)
model_small = SeCoModel(
    trace_dim=1024,
    out_dim=128,        # ← Spatial embedding dimension
    in_channels=7
)
# Output: [B, 128, 224, 224]
# Use case: Fast inference, edge deployment

# Option B: Default (256 channels)
model_default = SeCoModel(
    trace_dim=1024,
    out_dim=256,        # ← Default
    in_channels=7
)
# Output: [B, 256, 224, 224]
# Use case: General purpose, good balance

# Option C: Large (512 channels)
model_large = SeCoModel(
    trace_dim=1024,
    out_dim=512,        # ← More expressive
    in_channels=7
)
# Output: [B, 512, 224, 224]
# Use case: Global model, high diversity data


# Level 2: Projection Head Dimension (proj_dim)
# -----------------------------------------------
# Controls: [B, proj_dim] - Contrastive learning space (training only)
# Where: src/models.py, TRACE class, line ~120

# In src/models.py:
# make_head = lambda: nn.Sequential(
#     nn.Linear(dim, dim),
#     nn.GELU(),
#     nn.Linear(dim, 128),  # ← Change this to 64, 128, or 256
#     nn.BatchNorm1d(128),
# )

# Note: This only affects training. At inference, you use spatial embeddings.


# ============================================================================
# EXAMPLE: Training with custom dimensions
# ============================================================================

if __name__ == "__main__":
    import torch
    
    # Create model with custom out_dim
    model = SeCoModel(
        trace_dim=1024,
        out_dim=384,        # Custom: 384 channels
        in_channels=7
    )
    
    # Dummy data
    images = torch.randn(2, 5, 7, 224, 224)
    timesteps = torch.randint(0, 365, (2, 5))
    sensor_ids = torch.zeros(2, 5, dtype=torch.long)
    
    # Training: Get projection heads
    z0, z1, z2 = model(images, timesteps, sensor_ids, 
                       flavor="global", 
                       return_projections=True)
    print(f"Projection heads: {z0.shape}")  # [2, 128]
    
    # Inference: Get spatial embeddings
    emb = model.get_embeddings(images, timesteps, sensor_ids, flavor="global")
    print(f"Spatial embeddings: {emb.shape}")  # [2, 384, 224, 224]
    
    print(f"\n✓ Each pixel represents 10m x 10m")
    print(f"✓ Total coverage: 2.24km x 2.24km")
    print(f"✓ Embedding dimension: {emb.shape[1]} channels")
