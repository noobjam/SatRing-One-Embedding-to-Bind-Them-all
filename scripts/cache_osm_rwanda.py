"""
Preprocess and cache OSM data for all locations in Rwanda dataset.

This script downloads OSM land use tags for all tile locations and saves them
to disk for fast loading during training.

Usage:
    python scripts/cache_osm_rwanda.py --data-dir data/processed --output osm_cache_rwanda.pkl
"""

import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from osm_utils import get_osm_tags_for_location
from osm_tags import encode_osm_tag
from sequenceDataset import SequenceDataset

def parse_tile_location(filepath):
    """
    Extract lat/lon from tile filepath or location ID.
    
    Args:
        filepath: Path to .npy file or location tuple
    
    Returns:
        (lat, lon) or None
    """
    # This is dataset-specific. You'll need to extract coordinates from your file naming.
    # Placeholder implementation:
    
    # If your files are named like: S2_20230101_lat-1.5_lon29.5.npy
    # Parse the coordinates from filename
    
    # For now, return None (will skip OSM lookups)
    # TODO: Implement based on your file naming convention
    return None

def cache_osm_for_dataset(data_dir, output_path):
    """
    Download and cache OSM tags for all locations in dataset.
    
    Args:
        data_dir: Path to processed data directory
        output_path: Where to save the OSM cache (pickle file)
    """
    print("Loading dataset to find all locations...")
    dataset = SequenceDataset(
        processed_data=data_dir,
        sequence_length=5,
        augment=False
    )
    
    print(f"Found {len(dataset.locations)} unique locations")
    
    osm_cache = {}
    skipped = 0
    
    print("Downloading OSM tags for each location...")
    for loc in tqdm(dataset.locations, desc="Caching OSM"):
        # Extract coordinates from location ID
        coords = parse_tile_location(loc)
        
        if coords is None:
            # Can't get coordinates, skip
            osm_cache[loc] = 0  # Unknown
            skipped += 1
            continue
        
        lat, lon = coords
        
        # Query OSM for this location
        tags = get_osm_tags_for_location(lat, lon, radius=1000)
        
        # Encode to integer
        if tags and 'landuse' in tags:
            tag_str = tags['landuse']
        elif tags and 'natural' in tags:
            tag_str = tags['natural']
        else:
            tag_str = 'unknown'
        
        osm_encoded = encode_osm_tag(tag_str)
        osm_cache[loc] = osm_encoded
    
    # Save cache
    print(f"\nSaving OSM cache to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(osm_cache, f)
    
    print(f"✓ Cached OSM tags for {len(osm_cache)} locations")
    print(f"  - Skipped: {skipped} (no coordinates)")
    print(f"  - With tags: {len([v for v in osm_cache.values() if v > 0])}")
    
    # Show distribution
    tag_counts = {}
    for tag_int in osm_cache.values():
        tag_counts[tag_int] = tag_counts.get(tag_int, 0) + 1
    
    print("\nOSM Tag Distribution:")
    from osm_tags import decode_osm_tag
    for tag_int, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        tag_name = decode_osm_tag(tag_int)
        print(f"  {tag_name}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Cache OSM data for Rwanda dataset")
    parser.add_argument("--data-dir", type=str, required=True, help="Processed data directory")
    parser.add_argument("--output", type=str, default="osm_cache_rwanda.pkl", help="Output pickle file")
    args = parser.parse_args()
    
    cache_osm_for_dataset(args.data_dir, args.output)
    
    print("\n" + "="*80)
    print("Next steps:")
    print("1. Update SequenceDataset to load this cache:")
    print("   self.osm_cache = pickle.load(open('osm_cache_rwanda.pkl', 'rb'))")
    print("2. Modify _get_osm_tag() to return self.osm_cache.get(loc, 0)")
    print("3. Train model - the 4th loss (L_semantic) will now be active!")
    print("="*80)

if __name__ == "__main__":
    main()
