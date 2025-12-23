import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import entropy
from tqdm import tqdm


RWANDA_TILES = [
    '35MPT', '35MQU', '35MQS', '35MQT',
    '35MRS', '35MRT', '35MRU',
    '36MTC', '36MTD'
]

PATCH_SIZE_M = 2240

COMPLEXITY_BINS = [0, 0.5, 1.2, 5.0]
COMPLEXITY_LABELS = ['homogeneous', 'moderate', 'complex']

MIN_SAMPLES_PER_STRATUM = 300


def extract_tile_from_filename(filename):
    parts = filename.split('_')
    for part in parts:
        if part.startswith('T') and len(part) >= 5:
            return part[:6] if len(part) >= 6 else part
        if len(part) >= 5 and part[:2].isdigit():
            return part[:6] if len(part) >= 6 else part
    return "UNKNOWN"


def compute_land_cover_from_bands(patch_df):
    if 'B8' not in patch_df.columns or 'B4' not in patch_df.columns:
        return 'Other', 0.0, {}
    
    b8 = patch_df['B8'].to_numpy()
    b4 = patch_df['B4'].to_numpy()
    
    ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
    
    forest_pct = (ndvi > 0.6).sum() / len(ndvi)
    cropland_pct = ((ndvi > 0.3) & (ndvi <= 0.6)).sum() / len(ndvi)
    grassland_pct = ((ndvi > 0.15) & (ndvi <= 0.3)).sum() / len(ndvi)
    
    if 'B2' in patch_df.columns:
        b2 = patch_df['B2'].to_numpy()
        water_pct = (b2 < 0.05).sum() / len(b2)
    else:
        water_pct = 0.0
    
    if 'B11' in patch_df.columns:
        b11 = patch_df['B11'].to_numpy()
        urban_pct = ((b11 > 0.2) & (ndvi < 0.2)).sum() / len(b11)
    else:
        urban_pct = 0.0
    
    wetland_pct = ((ndvi > 0.3) & (ndvi < 0.6) & (b4 < 0.1)).sum() / len(ndvi)
    
    class_probs = {
        'Forest': forest_pct,
        'Cropland': cropland_pct,
        'Grassland': grassland_pct,
        'Water': water_pct,
        'Built_Up_Area': urban_pct,
        'Wetland': wetland_pct,
        'Other': max(0, 1 - (forest_pct + cropland_pct + grassland_pct + water_pct + urban_pct + wetland_pct))
    }
    
    dominant_class = max(class_probs.items(), key=lambda x: x[1])[0]
    
    probs = np.array([v for v in class_probs.values() if v > 0])
    ent = entropy(probs) if len(probs) > 0 else 0.0
    
    return dominant_class, ent, class_probs


def generate_more_patches(existing_csv, processed_dir, additional_count, sensor_type='S2'):
    print(f"Loading existing {existing_csv}...")
    existing_df = pd.read_csv(existing_csv)
    
    existing_tiles = existing_df['tile'].unique()
    print(f"Existing patches cover {len(existing_tiles)} tiles")
    
    processed_dir = Path(processed_dir)
    sensor_dir = processed_dir / sensor_type
    
    if not sensor_dir.exists():
        print(f"Error: {sensor_dir} does not exist")
        return
    
    print(f"\nScanning {sensor_type} tiles for additional coverage...")
    
    tile_files = {}
    for date_dir in sensor_dir.iterdir():
        if not date_dir.is_dir() or not date_dir.name.isdigit():
            continue
        
        for pq_file in date_dir.glob("*.parquet"):
            tile = extract_tile_from_filename(pq_file.name)
            
            if tile not in tile_files:
                tile_files[tile] = []
            tile_files[tile].append(pq_file)
    
    print(f"Found {len(tile_files)} tiles in processed data")
    
    new_candidates = []
    
    for tile in tqdm(existing_tiles, desc="Generating additional patches"):
        if tile not in tile_files:
            continue
        
        sample_file = tile_files[tile][0]
        
        try:
            df = pl.read_parquet(sample_file)
            
            if 'x' in df.columns and 'y' in df.columns:
                xs = df['x'].to_numpy()
                ys = df['y'].to_numpy()
            elif 'longitude' in df.columns and 'latitude' in df.columns:
                continue
            else:
                continue
            
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            
            existing_tile_patches = existing_df[existing_df['tile'] == tile]
            existing_coords = set(zip(existing_tile_patches['lon'], existing_tile_patches['lat']))
            
            step = PATCH_SIZE_M
            x_range = np.arange(x_min, x_max, step)
            y_range = np.arange(y_min, y_max, step)
            
            for x in x_range:
                for y in y_range:
                    x_center = x + step / 2
                    y_center = y + step / 2
                    
                    if (x_center, y_center) in existing_coords:
                        continue
                    
                    bbox = (x, y, x + step, y + step)
                    
                    try:
                        patch_df = df.filter(
                            (pl.col('x') >= bbox[0]) &
                            (pl.col('x') <= bbox[2]) &
                            (pl.col('y') >= bbox[1]) &
                            (pl.col('y') <= bbox[3])
                        )
                        
                        if len(patch_df) < 1000:
                            continue
                        
                        dominant_class, ent, class_dist = compute_land_cover_from_bands(patch_df)
                        
                        new_candidates.append({
                            'tile': tile,
                            'lon': x_center,
                            'lat': y_center,
                            'dominant_class': dominant_class,
                            'entropy': ent,
                            'class_distribution': class_dist
                        })
                    except:
                        continue
        except:
            continue
    
    if not new_candidates:
        print("No new candidates found")
        return None
    
    candidates_df = pd.DataFrame(new_candidates)
    print(f"\nFound {len(candidates_df)} new candidate patches")
    
    candidates_df['complexity'] = pd.cut(
        candidates_df['entropy'],
        bins=COMPLEXITY_BINS,
        labels=COMPLEXITY_LABELS
    )
    
    print("\nNew candidates distribution:")
    strata = candidates_df.groupby(['dominant_class', 'complexity']).size()
    for (cls, comp), count in strata.items():
        print(f"  {cls:20s} × {comp:12s}: {count:5d}")
    
    existing_strata = existing_df.groupby(['dominant_class', 'complexity']).size()
    
    sampled_new = []
    
    print(f"\nSampling up to {additional_count} additional patches to fill gaps...")
    
    for (cls, complexity), count in strata.items():
        existing_count = existing_strata.get((cls, complexity), 0)
        
        stratum_candidates = candidates_df[
            (candidates_df['dominant_class'] == cls) &
            (candidates_df['complexity'] == complexity)
        ]
        
        target = max(MIN_SAMPLES_PER_STRATUM - existing_count, 0)
        target = min(target, len(stratum_candidates))
        
        if target > 0:
            sampled = stratum_candidates.sample(n=target, random_state=42)
            sampled_new.append(sampled)
            print(f"  {cls:20s} × {complexity:12s}: +{target:4d} (was {existing_count:4d})")
    
    if not sampled_new:
        print("No additional patches needed - existing distribution is balanced")
        return None
    
    new_patches_df = pd.concat(sampled_new).head(additional_count)
    
    new_patches_df = new_patches_df[['tile', 'lon', 'lat', 'dominant_class', 'complexity', 'entropy']]
    
    combined_df = pd.concat([existing_df, new_patches_df], ignore_index=True)
    combined_df['patch_id'] = range(len(combined_df))
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate additional stratified patches to expand existing dataset"
    )
    parser.add_argument(
        '--existing-csv',
        type=str,
        default='notebooks/patches_rwanda_stratified.csv',
        help='Path to existing patches CSV'
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed',
        help='Path to preprocessed data directory'
    )
    parser.add_argument(
        '--additional-count',
        type=int,
        default=12000,
        help='Number of additional patches to generate (default: 12000 for total ~15K)'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default='patch_metadata/patches_rwanda_15k.csv',
        help='Output path for expanded patches CSV'
    )
    parser.add_argument(
        '--sensor',
        type=str,
        default='S2',
        choices=['S2', 'LS'],
        help='Sensor type to use'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("EXPANDING STRATIFIED PATCH DATASET")
    print("="*80)
    
    expanded_df = generate_more_patches(
        args.existing_csv,
        args.processed_dir,
        args.additional_count,
        args.sensor
    )
    
    if expanded_df is None:
        print("\nNo expansion needed or possible.")
        return
    
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    expanded_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print("EXPANSION COMPLETE")
    print('='*80)
    print(f"\nExpanded from {len(pd.read_csv(args.existing_csv))} to {len(expanded_df)} patches")
    print(f"Saved to: {output_path}")
    print("\nFinal distribution:")
    final_dist = expanded_df.groupby(['dominant_class', 'complexity']).size()
    for (cls, comp), count in final_dist.items():
        print(f"  {cls:20s} × {comp:12s}: {count:5d}")
    print('='*80 + "\n")


if __name__ == "__main__":
    main()
