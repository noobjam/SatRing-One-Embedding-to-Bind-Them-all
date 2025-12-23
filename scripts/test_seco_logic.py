import sys
from pathlib import Path
import shutil
import random
import pandas as pd
import polars as pl
import numpy as np
import torch
sys.path.append(str(Path.cwd()))
sys.path.append(str(Path.cwd() / "src"))
from src.seco_dataset import SeCoDataset
def create_mock_parquet(path, omn_val=0, size=2240, bands=12):
    # Determine grid size (assuming roughly 10m resolution)
    dim = size // 10
    total_pixels = dim * dim
    
    data = {
        'x': np.tile(np.arange(0, size, 10), dim),
        'y': np.repeat(np.arange(0, size, 10), dim),
        'OMN': np.full(total_pixels, omn_val, dtype=np.int32)
    }
    
    band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    for b in band_names:
        data[b] = np.random.rand(total_pixels).astype(np.float32)
        
    df = pl.DataFrame(data)
    df.write_parquet(path)

def test_seco_logic():
    # Setup paths
    base_dir = Path("test_data_seco")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir()
    
    tiles_dir = base_dir / "tiles"
    tiles_dir.mkdir()
    
    # Create mock patch CSV
    csv_path = base_dir / "test_patches.csv"
    patches_data = [{
        'tile': 'T01', 'lon': 1120, 'lat': 1120, 
        'dominant_class': 'Forest', 'complexity': 'low'
    }]
    pd.DataFrame(patches_data).to_csv(csv_path, index=False)
    
    # Create mock Tile directory T01
    t1_dir = tiles_dir / "T01"
    t1_dir.mkdir()
    
    # Frame 1: Clear (OMN=0) -> Date 20230101
    (t1_dir / "20230101").mkdir()
    create_mock_parquet(t1_dir / "20230101" / "part.parquet", omn_val=0)
    
    # Frame 2: Cloudy (OMN=2) -> Date 20230201 -> Should be REJECTED
    (t1_dir / "20230201").mkdir()
    create_mock_parquet(t1_dir / "20230201" / "part.parquet", omn_val=2)
    
    # Frame 3: Clear (OMN=0) -> Date 20230301
    (t1_dir / "20230301").mkdir()
    create_mock_parquet(t1_dir / "20230301" / "part.parquet", omn_val=0)
    
    print("\n--- Testing SeCoDataset Logic ---")
    
    # Init Dataset
    dataset = SeCoDataset(
        tiles_dir=str(tiles_dir),
        patches_csv=str(csv_path),
        patch_size=224,
        patch_size_m=2240
    )
    
    print(f"\nNum valid patches: {len(dataset)}")
    
    if len(dataset) == 0:
        print("FAIL: No valid patches found.")
        return

    # Check that Cloudy frame was rejected
    patch_item = dataset.valid_patches[0]
    frame_dates = sorted([f['date'] for f in patch_item['frames']])
    print(f"Accepted Dates: {frame_dates}")
    
    if '20230201' in frame_dates:
        print("FAIL: Cloudy frame (20230201) was NOT accepted.")
    else:
        print("PASS: Cloudy frame correctly rejected.")
         
    # Test Pair Creation (Snapshot)
    print("\n--- Verification of 4-Subspace Pairs (Snapshot Mode) ---")
    sample = dataset[0]
    
    # Check structure
    v1 = sample['view1']
    v2 = sample['view2']
    print(f"Image Shape: {v1['image'].shape}") # [T=1, C, H, W]
    
    if v1['image'].shape[0] == 1:
        print("PASS: Snapshot mode returns T=1.")
    else:
        print(f"FAIL: Expected T=1, got {v1['image'].shape[0]}")
        
    # Re-init for Sequence Mode
    print("\n--- Verification of Sequence Mode (Len=2) ---")
    dataset_seq = SeCoDataset(
        tiles_dir=str(tiles_dir),
        patches_csv=str(csv_path),
        patch_size=224,
        patch_size_m=2240,
        sequence_length=2
    )
    sample_seq = dataset_seq[0]
    seq1 = sample_seq['view1']
    
    print(f"Seq Image Shape: {seq1['image'].shape}") # [T=2, C, H, W]
    if seq1['image'].shape[0] == 2:
        print("PASS: Sequence mode returns T=2.")
    else:
        print(f"FAIL: Expected T=2, got {seq1['image'].shape[0]}")
        
    # Test Full Sequence Mode (Padding)
    print("\n--- Verification of Full Sequence Mode (Len=0, Max=5) ---")
    dataset_full = SeCoDataset(
        tiles_dir=str(tiles_dir),
        patches_csv=str(csv_path),
        patch_size=224,
        patch_size_m=2240,
        sequence_length=0, # Full mode
        max_sequence_length=5 # Test buffer
    )
    sample_full = dataset_full[0]
    full1 = sample_full['view1']
    
    # We have 2 mock frames (Clear: 20230101, 20230301). 
    # Buffer is 5. Expected: 2 valid frames + 3 padded frames.
    
    print(f"Full Image Shape: {full1['image'].shape}") # [5, 12, 224, 224]
    if full1['image'].shape[0] == 5:
        print("PASS: Padded sequence length is Max=5.")
    else:
        print(f"FAIL: Expected T=5, got {full1['image'].shape[0]}")
        
    mask = full1['mask']
    print(f"Mask: {mask}")
    if mask.sum() == 2:
        print("PASS: Mask identifies 2 valid frames.")
    else:
        print(f"FAIL: Expected 2 valid frames, got {mask.sum()}")
    
    # Verify padding is zeros
    if full1['image'][2:].sum() == 0:
         print("PASS: Padding is zero.")
    else:
         print("FAIL: Padding area contains non-zeros.")

    # Test Random Mixed Mode (-1)
    print("\n--- Verification of Random Mixed Mode (Len=-1) ---")
    dataset_mix = SeCoDataset(
        tiles_dir=str(tiles_dir),
        patches_csv=str(csv_path),
        patch_size=224,
        patch_size_m=2240,
        sequence_length=-1, # Random Mix
        max_sequence_length=5
    )
    # We run multiple samples to check variability
    print("Sampling multiple mixed pairs...")
    variability = False
    for i in range(10):
        s = dataset_mix[0]
        len1 = s['view1']['mask'].sum().item()
        len2 = s['view2']['mask'].sum().item()
        print(f"Sample {i}: L1={len1}, L2={len2}")
        
        if len1 != len2:
            variability = True
            
    if variability:
        print("PASS: Observed different lengths between view1/view2 or across samples.")
    else:
        print("WARN: No variability observed (might be due to small mock dataset size).")

    # Cleanup
    shutil.rmtree(base_dir)
    print("\nTest Complete.")

if __name__ == "__main__":
    test_seco_logic()
