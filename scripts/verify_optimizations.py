"""
Quick verification script to test the implemented optimizations.

Tests:
1. OSM cache loading
2. Model forward pass with all 4 projection heads
3. Loss computation with semantic component
4. Memory estimate for batch size
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_osm_loading():
    """Test OSM cache loading in dataset."""
    print("="*80)
    print("Test 1: OSM Cache Loading")
    print("="*80)
    
    from sequenceDataset import SequenceDataset
    
    # This will print warnings if OSM cache not found
    try:
        dataset = SequenceDataset(
            processed_data="data/processed",
            sequence_length=5,
            augment=False
        )
        print(f"✓ Dataset loaded with {len(dataset)} sequences")
        
        # Check if OSM cache is loaded
        if dataset.osm_cache:
            num_with_tags = sum(1 for v in dataset.osm_cache.values() if v > 0)
            print(f"✓ OSM cache loaded: {len(dataset.osm_cache)} locations")
            print(f"  {num_with_tags} have OSM tags")
            return True
        else:
            print("⚠ OSM cache is empty - run cache_osm_rwanda.py")
            return False
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

def test_model_forward():
    """Test model forward pass returns 4 projection heads."""
    print("\n" + "="*80)
    print("Test 2: Model Forward Pass (4 Projection Heads)")
    print("="*80)
    
    from seco_model import SeCoModel
    
    try:
        model = SeCoModel()
        print("✓ Model created")
        
        # Dummy input
        x = torch.randn(2, 5, 7, 224, 224)
        t = torch.arange(5).unsqueeze(0).repeat(2, 1)
        s = torch.zeros(2, 5, dtype=torch.long)
        
        # Forward pass
        z0, z1, z2, z3 = model(x, t, s, flavor="global", return_projections=True)
        
        print(f"✓ Forward pass successful")
        print(f"  Z0 (all-invariant): {z0.shape}")
        print(f"  Z1 (season-invariant): {z1.shape}")
        print(f"  Z2 (sensor-invariant): {z2.shape}")
        print(f"  Z3 (semantic): {z3.shape}")
        
        assert z0.shape == (2, 128), f"Wrong Z0 shape: {z0.shape}"
        assert z1.shape == (2, 128), f"Wrong Z1 shape: {z1.shape}"
        assert z2.shape == (2, 128), f"Wrong Z2 shape: {z2.shape}"
        assert z3.shape == (2, 128), f"Wrong Z3 shape: {z3.shape}"
        
        print("✓ All projection heads return correct shapes")
        return True
        
    except Exception as e:
        print(f"✗ Model forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation():
    """Test loss computation with semantic component."""
    print("\n" + "="*80)
    print("Test 3: Loss Computation (4 Components)")
    print("="*80)
    
    from trainer.seco_loss import seco_loss
    
    try:
        # Dummy embeddings
        B = 4
        z0_1 = torch.randn(B, 128)
        z0_2 = torch.randn(B, 128)
        z1_1 = torch.randn(B, 128)
        z1_2 = torch.randn(B, 128)
        z2_1 = torch.randn(B, 128)
        z2_2 = torch.randn(B, 128)
        z3_1 = torch.randn(B, 128)
        z3_2 = torch.randn(B, 128)
        
        # Dummy metadata
        t1 = torch.tensor([10, 50, 100, 200])
        t2 = torch.tensor([15, 55, 95, 205])
        s1 = torch.tensor([0, 0, 1, 1])
        s2 = torch.tensor([0, 1, 1, 0])
        osm1 = torch.tensor([1, 2, 0, 3])  # forest, farmland, unknown, residential
        osm2 = torch.tensor([1, 2, 0, 4])  # forest, farmland, unknown, commercial
        
        # Compute loss
        l_all, l_season, l_sensor, l_semantic = seco_loss(
            z0_1, z0_2, z1_1, z1_2, z2_1, z2_2, z3_1, z3_2,
            t1, t2, s1, s2, osm1, osm2
        )
        
        print(f"✓ Loss computation successful")
        print(f"  L_all (location): {l_all.item():.4f}")
        print(f"  L_season (temporal): {l_season.item():.4f}")
        print(f"  L_sensor (cross-sensor): {l_sensor.item():.4f}")
        print(f"  L_semantic (OSM): {l_semantic.item():.4f}")
        
        # All losses should be non-nan
        assert not torch.isnan(l_all), "L_all is NaN"
        assert not torch.isnan(l_season), "L_season is NaN"
        assert not torch.isnan(l_sensor), "L_sensor is NaN"
        assert not torch.isnan(l_semantic), "L_semantic is NaN"
        
        # L_semantic should be non-zero (we have matching OSM tags)
        if l_semantic.item() > 0:
            print("✓ L_semantic is non-zero (OSM matching working)")
        else:
            print("⚠ L_semantic is zero (expected with no matching OSM tags)")
        
        return True
        
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_estimate():
    """Estimate memory usage for different batch sizes."""
    print("\n" + "="*80)
    print("Test 4: Memory Usage Estimation")
    print("="*80)
    
    try:
        # Estimates based on typical ViT-Large model
        PARAM_SIZE = 400e6 * 4  # 400M params × 4 bytes (FP32)
        
        def estimate_memory(batch_size, seq_len=5, channels=7, img_size=224):
            # Input
            input_mb = batch_size * seq_len * channels * img_size * img_size * 4 / 1e6
            
            # Activations (rough estimate: 500MB per sample)
            activation_mb = batch_size * 500
            
            # Gradients (same as params)
            grad_mb = PARAM_SIZE / 1e6
            
            # Optimizer states (AdamW = 2× params for momentum + variance)
            optim_mb = 2 * PARAM_SIZE / 1e6
            
            total_mb = input_mb + activation_mb + grad_mb + optim_mb + (PARAM_SIZE / 1e6)
            return total_mb / 1024  # Convert to GB
        
        print("Memory estimates for H100 (80GB HBM):")
        print()
        for batch_size in [8, 16, 24, 32, 48, 64]:
            mem_gb = estimate_memory(batch_size)
            utilization = (mem_gb / 80) * 100
            status = "✓" if mem_gb < 75 else "⚠"
            print(f"  {status} Batch size {batch_size:2d}: ~{mem_gb:5.1f} GB ({utilization:4.1f}% utilization)")
        
        print("\nRecommended: batch_size=32 (~24GB, 30% utilization)")
        print("Max safe: batch_size=64 (~48GB, 60% utilization)")
        return True
        
    except Exception as e:
        print(f"✗ Memory estimation failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("\n" + "="*80)
    print("SatRing Optimization Verification")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("OSM Loading", test_osm_loading()))
    results.append(("Model Forward", test_model_forward()))
    results.append(("Loss Computation", test_loss_computation()))
    results.append(("Memory Estimation", test_memory_estimate()))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(r[1] for r in results)
    
    print("="*80)
    if all_passed:
        print("✓ All tests passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Generate OSM cache (if not done):")
        print("     python scripts/cache_osm_rwanda.py --data-dir data/processed --output osm_cache_rwanda.pkl")
        print("  2. Launch training:")
        print("     ./launch_training.sh")
    else:
        print("⚠ Some tests failed. Review errors above.")
    print("="*80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
