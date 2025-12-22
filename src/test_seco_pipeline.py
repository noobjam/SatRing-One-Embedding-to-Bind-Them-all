import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Mock torchgeo
from unittest.mock import MagicMock
sys.modules["torchgeo"] = MagicMock()
sys.modules["torchgeo.models"] = MagicMock()
# Mock DOFA weights and model
mock_dofa = MagicMock()

# Make patch_embed return correct batch size
def mock_patch_embed(x, wavelengths):
    B = x.shape[0]
    return (torch.randn(B, 196, 1024), None)

mock_dofa.patch_embed = mock_patch_embed 
mock_dofa.cls_token = torch.randn(1, 1, 1024)
mock_dofa.pos_embed = torch.randn(1, 197, 1024)
mock_dofa.blocks = nn.ModuleList([nn.Identity()])
mock_dofa.fc_norm = nn.Identity()

def mock_dofa_fn(*args, **kwargs):
    return mock_dofa

sys.modules["torchgeo.models"].dofa_large_patch16_224 = mock_dofa_fn
sys.modules["torchgeo.models"].DOFALarge16_Weights = MagicMock()

from seco_model import SeCoModel
from trainer.seco_loss import seco_loss

def test_pipeline():
    print("Initializing SeCoModel...")
    # Use dim=1024 to match DOFA Large
    model = SeCoModel(trace_dim=1024, out_dim=256, in_channels=7)
    
    B, T, C, H, W = 2, 5, 7, 224, 224
    print(f"Creating dummy data: [{B}, {T}, {C}, {H}, {W}]")
    
    v1_img = torch.randn(B, T, C, H, W)
    v1_t = torch.randint(0, 365, (B, T))
    v1_s = torch.randint(0, 2, (B, T))
    
    v2_img = torch.randn(B, T, C, H, W)
    v2_t = torch.randint(0, 365, (B, T))
    v2_s = torch.randint(0, 2, (B, T))
    
    print("Forward pass View 1...")
    z0_1, z1_1, z2_1 = model(v1_img, v1_t, v1_s)
    print(f"Output shapes: z0={z0_1.shape}, z1={z1_1.shape}, z2={z2_1.shape}")
    
    print("Forward pass View 2...")
    z0_2, z1_2, z2_2 = model(v2_img, v2_t, v2_s)
    
    print("Computing Loss...")
    l_all, l_season, l_sensor = seco_loss(
        z0_1, z0_2, z1_1, z1_2, z2_1, z2_2,
        v1_t, v2_t, v1_s, v2_s
    )
    
    print(f"Losses: All={l_all.item()}, Season={l_season.item()}, Sensor={l_sensor.item()}")
    print("Test Passed!")

if __name__ == "__main__":
    test_pipeline()
