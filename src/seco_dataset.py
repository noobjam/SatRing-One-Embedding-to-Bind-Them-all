import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler

from config import config


CLOUD_THRESHOLD_PATCH = 0.70
USABLE_THRESHOLD_PATCH = 0.15
MIN_CLEAR_FRAMES = 2
FRAME_CLEAR_THRESHOLD = 0.20

CLOUD_BINS = [0, 0.20, 0.50, 0.70, 1.0]
CLOUD_LABELS = ['clear', 'partial', 'cloudy', 'rejected']
CLOUD_WEIGHTS = {'clear': 0.60, 'partial': 0.30, 'cloudy': 0.10}


class SeCoDataset(Dataset):
    def __init__(
        self,
        processed_dir: str,
        patches_csv: str,
        osm_cache_path: str = None,
        patch_size: int = 224,
        patch_size_m: int = 2240,
        sequence_length: int = 5,
        mode: str = "both",
        augment: bool = True,
        sensor_types: list = None
    ):
        self.processed_dir = Path(processed_dir)
        self.patch_size = patch_size
        self.patch_size_m = patch_size_m
        self.sequence_length = sequence_length
        self.mode = mode
        self.augment = augment
        self.sensor_types = sensor_types if sensor_types else ["S2", "LS"]
        
        self.sensor_mapping = {"S2": 0, "LS": 1, "S1": 2}
        
        print(f"Loading patches from {patches_csv}...")
        self.patches_df = pd.read_csv(patches_csv)
        
        if osm_cache_path and Path(osm_cache_path).exists():
            with open(osm_cache_path, 'rb') as f:
                self.osm_cache = pickle.load(f)
            print(f"Loaded OSM cache: {len(self.osm_cache)} locations")
        else:
            self.osm_cache = {}
            print("No OSM cache loaded")
        
        self._build_tile_index()
        self._filter_and_index_patches()
        self._compute_sampling_weights()
        
        print(f"Dataset ready: {len(self.valid_patches)} patches")

    def _build_tile_index(self):
        self.tile_index = defaultdict(lambda: defaultdict(list))
        
        for sensor_type in self.sensor_types:
            sensor_dir = self.processed_dir / sensor_type
            if not sensor_dir.exists():
                continue
            
            for date_dir in sensor_dir.iterdir():
                if not date_dir.is_dir() or not date_dir.name.isdigit():
                    continue
                
                date = date_dir.name
                parquet_files = list(date_dir.glob("*.parquet"))
                
                for pq_file in parquet_files:
                    tile_name = self._extract_tile_from_filename(pq_file.name)
                    self.tile_index[sensor_type][date].append({
                        'file': pq_file,
                        'tile': tile_name
                    })
        
        total_files = sum(
            len(dates) 
            for sensor in self.tile_index.values() 
            for dates in sensor.values()
        )
        print(f"Indexed {total_files} parquet files")

    def _extract_tile_from_filename(self, filename):
        parts = filename.split('_')
        for part in parts:
            if part.startswith('T') and len(part) == 6:
                return part
            if len(part) == 6 and part[:3].isdigit():
                return part
        return "UNKNOWN"

    def _filter_and_index_patches(self):
        print("Filtering patches by cloud quality...")
        
        valid_patches = []
        
        for idx, patch in self.patches_df.iterrows():
            patch_data = self._load_patch_temporal_data(patch)
            
            if patch_data is None:
                continue
            
            cloud_stats = self._compute_patch_cloud_stats(patch_data)
            
            if not self._passes_cloud_quality_gates(cloud_stats):
                continue
            
            cloud_category = pd.cut(
                [cloud_stats['median_cloud_pct']],
                bins=CLOUD_BINS,
                labels=CLOUD_LABELS
            )[0]
            
            if cloud_category == 'rejected':
                continue
            
            valid_patches.append({
                'patch_id': idx,
                'patch_info': patch,
                'patch_data': patch_data,
                'cloud_stats': cloud_stats,
                'cloud_category': cloud_category
            })
        
        self.valid_patches = valid_patches
        print(f"Retained {len(valid_patches)} / {len(self.patches_df)} patches")
        
        cloud_dist = pd.Series([p['cloud_category'] for p in valid_patches]).value_counts()
        print("Cloud distribution:")
        for cat, count in cloud_dist.items():
            print(f"  {cat}: {count}")

    def _load_patch_temporal_data(self, patch):
        lon, lat = patch['lon'], patch['lat']
        tile = patch['tile']
        
        bbox = self._compute_bbox(lon, lat)
        
        patch_data = []
        
        for sensor_type in self.sensor_types:
            if sensor_type not in self.tile_index:
                continue
            
            for date, files_info in self.tile_index[sensor_type].items():
                matching_files = [
                    f['file'] for f in files_info 
                    if f['tile'] == tile
                ]
                
                if not matching_files:
                    continue
                
                for pq_file in matching_files:
                    try:
                        patch_df = self._extract_patch_from_parquet(
                            pq_file, bbox
                        )
                        
                        if patch_df is None or len(patch_df) < 100:
                            continue
                        
                        cloud_pct = self._compute_cloud_percentage(patch_df)
                        usable_pct = self._compute_usable_percentage(patch_df)
                        
                        if cloud_pct > CLOUD_THRESHOLD_PATCH:
                            continue
                        if usable_pct < USABLE_THRESHOLD_PATCH:
                            continue
                        
                        patch_data.append({
                            'date': date,
                            'sensor': sensor_type,
                            'df': patch_df,
                            'cloud_pct': cloud_pct,
                            'usable_pct': usable_pct
                        })
                    except Exception as e:
                        continue
        
        if len(patch_data) < MIN_CLEAR_FRAMES:
            return None
        
        return patch_data

    def _compute_bbox(self, lon_utm, lat_utm):
        half_size = self.patch_size_m / 2
        return (
            lon_utm - half_size,
            lat_utm - half_size,
            lon_utm + half_size,
            lat_utm + half_size
        )

    def _extract_patch_from_parquet(self, pq_file, bbox):
        try:
            df = pl.read_parquet(pq_file)
            
            patch_df = df.filter(
                (pl.col('longitude') >= bbox[0]) &
                (pl.col('longitude') <= bbox[2]) &
                (pl.col('latitude') >= bbox[1]) &
                (pl.col('latitude') <= bbox[3])
            )
            
            return patch_df
        except Exception:
            return None

    def _compute_cloud_percentage(self, patch_df):
        if 'OMN' not in patch_df.columns:
            return 0.0
        
        omn = patch_df['OMN'].to_numpy()
        cloud_pixels = np.isin(omn, [1, 2]).sum()
        return cloud_pixels / len(omn) if len(omn) > 0 else 0.0

    def _compute_usable_percentage(self, patch_df):
        if 'OMN' not in patch_df.columns:
            return 1.0
        
        omn = patch_df['OMN'].to_numpy()
        clear_pixels = (omn == 0).sum()
        return clear_pixels / len(omn) if len(omn) > 0 else 0.0

    def _compute_patch_cloud_stats(self, patch_data):
        cloud_pcts = [p['cloud_pct'] for p in patch_data]
        clear_frames = sum(1 for p in patch_data if p['cloud_pct'] < FRAME_CLEAR_THRESHOLD)
        
        return {
            'median_cloud_pct': np.median(cloud_pcts),
            'mean_cloud_pct': np.mean(cloud_pcts),
            'clear_frame_count': clear_frames,
            'total_frames': len(patch_data)
        }

    def _passes_cloud_quality_gates(self, cloud_stats):
        if cloud_stats['clear_frame_count'] < MIN_CLEAR_FRAMES:
            return False
        
        if cloud_stats['median_cloud_pct'] > CLOUD_THRESHOLD_PATCH:
            return False
        
        return True

    def _compute_sampling_weights(self):
        weights = []
        
        for patch_meta in self.valid_patches:
            complexity = patch_meta['patch_info']['complexity']
            dominant_class = patch_meta['patch_info']['dominant_class']
            cloud_category = patch_meta['cloud_category']
            
            class_weight = 1.0
            complexity_weight = 1.0
            cloud_weight = CLOUD_WEIGHTS.get(cloud_category, 0.1)
            
            combined_weight = class_weight * complexity_weight * cloud_weight
            weights.append(combined_weight)
        
        self.sampling_weights = torch.tensor(weights, dtype=torch.float)
        print(f"Computed sampling weights: min={self.sampling_weights.min():.3f}, max={self.sampling_weights.max():.3f}")

    def get_sampler(self):
        return WeightedRandomSampler(
            weights=self.sampling_weights,
            num_samples=len(self.valid_patches),
            replacement=True
        )

    def __len__(self):
        return len(self.valid_patches)

    def __getitem__(self, idx):
        patch_meta = self.valid_patches[idx]
        patch_data = patch_meta['patch_data']
        patch_info = patch_meta['patch_info']
        
        v1_data = self._sample_sequence(patch_data)
        v2_data = self._sample_sequence(patch_data)
        
        v1_img, v1_t, v1_s = self._process_sequence(v1_data, augment=self.augment)
        v2_img, v2_t, v2_s = self._process_sequence(v2_data, augment=self.augment)
        
        osm_tag = self.osm_cache.get(patch_meta['patch_id'], 0)
        
        return {
            'v1_img': v1_img,
            'v1_t': v1_t,
            'v1_s': v1_s,
            'v2_img': v2_img,
            'v2_t': v2_t,
            'v2_s': v2_s,
            'osm_tag': torch.tensor(osm_tag, dtype=torch.long),
            'dominant_class': patch_info['dominant_class'],
            'complexity': patch_info['complexity'],
            'cloud_category': patch_meta['cloud_category'],
            'loc_idx': idx
        }

    def _sample_sequence(self, patch_data):
        clear_frames = [p for p in patch_data if p['cloud_pct'] < FRAME_CLEAR_THRESHOLD]
        
        if len(clear_frames) >= self.sequence_length:
            sampled = random.sample(clear_frames, self.sequence_length)
        else:
            sampled = clear_frames + random.sample(
                patch_data,
                self.sequence_length - len(clear_frames)
            )
        
        sampled.sort(key=lambda x: x['date'])
        return sampled

    def _process_sequence(self, sequence_data, augment=False):
        images = []
        timestamps = []
        sensors = []
        
        for frame in sequence_data:
            img = self._df_to_tensor(frame['df'])
            
            date_str = frame['date']
            doy = pd.to_datetime(date_str, format='%Y%m%d').dayofyear
            
            sensor_id = self.sensor_mapping.get(frame['sensor'], 0)
            
            images.append(img)
            timestamps.append(doy)
            sensors.append(sensor_id)
        
        seq_tensor = torch.stack(images)
        
        if augment:
            seq_tensor = self._apply_augmentations(seq_tensor)
        
        return (
            seq_tensor,
            torch.tensor(timestamps, dtype=torch.long),
            torch.tensor(sensors, dtype=torch.long)
        )

    def _df_to_tensor(self, patch_df):
        band_cols = [c for c in patch_df.columns if c.startswith('B') and c != 'B8A']
        
        band_data = []
        for band in sorted(band_cols):
            values = patch_df[band].to_numpy().astype(np.float32)
            
            img = values.reshape(self.patch_size, self.patch_size)
            band_data.append(img)
        
        return torch.tensor(np.stack(band_data), dtype=torch.float32)

    def _apply_augmentations(self, seq_tensor):
        if random.random() > 0.5:
            seq_tensor = torch.flip(seq_tensor, dims=[-1])
        if random.random() > 0.5:
            seq_tensor = torch.flip(seq_tensor, dims=[-2])
        
        k = random.randint(0, 3)
        seq_tensor = torch.rot90(seq_tensor, k=k, dims=(-2, -1))
        
        noise = torch.randn_like(seq_tensor) * 0.01
        seq_tensor = seq_tensor + noise
        
        return seq_tensor
