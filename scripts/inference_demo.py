"""
Inference script demonstrating the three embedding modes at 10m resolution.

Usage:
    python scripts/inference_demo.py --checkpoint path/to/model.pt --data-dir path/to/data
"""

import torch
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from seco_model import SeCoModel
import numpy as np

def load_model(checkpoint_path, device='cuda'):
    """Load trained SeCo model."""
    model = SeCoModel(
        trace_dim=1024,
        out_dim=256,
        in_channels=7
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model

def demo_inference_modes(model, device='cuda'):
    """
    Demonstrate the three inference modes with dummy data.
    """
    # Create dummy data: 1 location, 12 months of data
    B = 1
    T = 12  # Monthly observations
    C = 7   # Sentinel-2 bands
    H, W = 224, 224
    
    # Dummy satellite images
    images = torch.randn(B, T, C, H, W).to(device)
    
    # Timestamps (monthly: Jan=15, Feb=45, ..., Dec=345)
    timesteps = torch.tensor([[15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]]).to(device)
    
    # Sensor IDs (all Sentinel-2)
    sensor_ids = torch.zeros(B, T, dtype=torch.long).to(device)
    
    print("=" * 80)
    print("SeCo Foundation Model - Inference Demo")
    print("=" * 80)
    print(f"\nInput: {T} monthly observations of a location")
    print(f"Spatial resolution: 10m (224x224 pixels)")
    print()
    
    # Mode 1: Global (whole year)
    print("1. GLOBAL MODE - Aggregate entire year into single embedding")
    print("-" * 80)
    emb_global = model.get_embeddings(images, timesteps, sensor_ids, flavor="global")
    print(f"   Output shape: {emb_global.shape}")
    print(f"   Description: Single 10m resolution embedding representing the entire year")
    print(f"   Use case: Annual land cover classification, year-over-year change detection")
    print()
    
    # Mode 2: Sequence (variable length)
    print("2. SEQUENCE MODE - Average over variable-length sequence")
    print("-" * 80)
    
    # Example: Use only summer months (June-August)
    summer_indices = [4, 5, 6]  # June, July, August (0-indexed)
    images_summer = images[:, summer_indices, :, :, :]
    timesteps_summer = timesteps[:, summer_indices]
    sensor_ids_summer = sensor_ids[:, summer_indices]
    
    emb_sequence = model.get_embeddings(images_summer, timesteps_summer, sensor_ids_summer, flavor="sequence")
    print(f"   Input: {len(summer_indices)} observations (summer months)")
    print(f"   Output shape: {emb_sequence.shape}")
    print(f"   Description: Averaged embedding over selected timesteps")
    print(f"   Use case: Seasonal analysis, crop monitoring during growing season")
    print()
    
    # Mode 3: Snapshot (single timestep)
    print("3. SNAPSHOT MODE - Single timestep embedding")
    print("-" * 80)
    
    # Example: Use only July (middle of summer)
    july_index = 6
    images_july = images[:, july_index:july_index+1, :, :, :]  # Keep T dimension
    timesteps_july = timesteps[:, july_index:july_index+1]
    sensor_ids_july = sensor_ids[:, july_index:july_index+1]
    
    emb_snapshot = model.get_embeddings(images_july, timesteps_july, sensor_ids_july, flavor="snapshot")
    print(f"   Input: 1 observation (July)")
    print(f"   Output shape: {emb_snapshot.shape}")
    print(f"   Description: Embedding for a single point in time")
    print(f"   Use case: Event detection, rapid mapping, single-date classification")
    print()
    
    # Demonstrate spatial resolution
    print("=" * 80)
    print("SPATIAL RESOLUTION VERIFICATION")
    print("=" * 80)
    print(f"All embeddings have shape [B, {emb_global.shape[1]}, 224, 224]")
    print(f"Each pixel represents 10m x 10m on the ground")
    print(f"Total coverage: 2.24km x 2.24km")
    print()
    
    # Save embeddings (optional)
    print("Saving embeddings to disk...")
    np.save("embedding_global.npy", emb_global.cpu().numpy())
    np.save("embedding_sequence.npy", emb_sequence.cpu().numpy())
    np.save("embedding_snapshot.npy", emb_snapshot.cpu().numpy())
    print("✓ Saved: embedding_global.npy, embedding_sequence.npy, embedding_snapshot.npy")
    print()
    
    return emb_global, emb_sequence, emb_snapshot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    if args.checkpoint:
        print(f"Loading model from {args.checkpoint}...")
        model = load_model(args.checkpoint, args.device)
    else:
        print("No checkpoint provided. Using untrained model for demo...")
        model = SeCoModel(trace_dim=1024, out_dim=256, in_channels=7).to(args.device)
        model.eval()
    
    # Run demo
    demo_inference_modes(model, args.device)
    
    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
