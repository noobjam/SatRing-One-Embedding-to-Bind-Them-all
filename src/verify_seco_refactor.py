import os
import numpy as np
import torch
from pathlib import Path
import shutil

# Import the refactored SeCoDataset
from seco_dataset import SeCoDataset

def setup_dummy_patches(base_dir):
    """
    Creates a mockup directory:
    base_dir / bin / tile / patch_{id}_{date}.npy
    """
    bins = ['0-10', '10-30']
    tile = 'T35MPT'
    patch_id = 'P001'
    dates = ['20240101', '20240201', '20240301']
    
    os.makedirs(base_dir, exist_ok=True)
    
    for i, date in enumerate(dates):
        # alternate bins
        bin_name = bins[i % len(bins)]
        tile_dir = Path(base_dir) / bin_name / tile
        tile_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy [13, 224, 224] array
        dummy_patch = np.random.rand(13, 224, 224).astype(np.float32)
        np.save(tile_dir / f"patch_{patch_id}_{date}.npy", dummy_patch)
        
    print(f"Dummy patches created in {base_dir}")

def test_dataset(base_dir):
    try:
        ds = SeCoDataset(patches_dir=base_dir, sequence_length=1)
        print(f"Dataset length: {len(ds)}")
        
        if len(ds) > 0:
            sample = ds[0]
            print("Successfully sampled item")
            print(f"View 1 image shape: {sample['view1']['image'].shape}")
            print(f"View 1 time: {sample['view1']['time']}")
            print(f"Tile: {sample['tile']}, Patch ID: {sample['patch_id']}")
            
            # Check if it returns 12 bands as expected
            assert sample['view1']['image'].shape == (1, 12, 224, 224)
            print("Shape validation passed!")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dir = "/tmp/seco_test_patches"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    setup_dummy_patches(test_dir)
    test_dataset(test_dir)
    
    # Cleanup
    shutil.rmtree(test_dir)
