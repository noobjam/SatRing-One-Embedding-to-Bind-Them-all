import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict
import random

class SequenceDataset(Dataset):
    def __init__(self, processed_data: str, sensor_type: str = 'fusion', sequence_length: int = 5, augment=True):
        self.sequence_length = sequence_length
        self.processed_data = Path(processed_data) / sensor_type
        self.augment = augment
        self.location_map = defaultdict(list)
        self._load_files()
        self.sequences = self._create_sequences()
        self.locations = list(self.location_map.keys())
        
        # Load OSM cache if available
        self.osm_cache = {}
        osm_cache_path = Path(processed_data).parent / 'osm_cache_rwanda.pkl'
        if osm_cache_path.exists():
            import pickle
            with open(osm_cache_path, 'rb') as f:
                self.osm_cache = pickle.load(f)
            print(f"✓ Loaded OSM cache with {len(self.osm_cache)} locations")
            num_with_tags = sum(1 for v in self.osm_cache.values() if v > 0)
            print(f"  {num_with_tags} locations have OSM tags")
        else:
            print(f"⚠ Warning: OSM cache not found at {osm_cache_path}")
            print(f"  L_semantic will be 0. Run 'python scripts/cache_osm_rwanda.py' to fix.")

    def _load_files(self):
        for product in self.processed_data.iterdir():
            if not product.is_dir():
                continue
            for pf in product.rglob('*.npy'):
                name = pf.stem
                try:
                    sensor, frame, i, j = name.split('_')
                    frame, i, j = int(frame), int(i), int(j)
                    self.location_map[(i, j)].append((frame, sensor, pf))
                except ValueError:
                    continue

    def _create_sequences(self):
        sequences = []
        for loc, frames in self.location_map.items():
            frames = sorted(frames, key=lambda x: x[0])
            for i in range(len(frames) - self.sequence_length + 1):
                seq = [f for _, _, f in frames[i:i+self.sequence_length]]
                sequences.append((loc, seq))
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Always return two views of the same location (Positive Pair for 'All')
        # The loss function will decide if they are valid for 'Season' or 'Sensor' based on metadata.
        
        loc, seq_files = self.sequences[idx]
        
        # View 1 (Anchor)
        # We use the sequence defined in self.sequences
        v1_tensor, v1_times, v1_sensors = self._load_sequence(seq_files)
        
        # View 2 (Positive)
        # Sample another sequence from the SAME location
        # It could be the same sequence (augmented), or a different time, or different sensor
        v2_tensor, v2_times, v2_sensors = self._sample_positive_view(loc)
        
        # Extract OSM tag for this location
        # For simplicity, we'll cache OSM tags by location
        # In practice, you'd precompute this and store it
        osm_tag = self._get_osm_tag(loc)
        
        return {
            "v1_img": v1_tensor,
            "v1_t": torch.tensor(v1_times, dtype=torch.long),
            "v1_s": torch.tensor(v1_sensors, dtype=torch.long),
            "v1_osm": osm_tag,  # NEW: OSM land use tag (int)
            "v2_img": v2_tensor,
            "v2_t": torch.tensor(v2_times, dtype=torch.long),
            "v2_s": torch.tensor(v2_sensors, dtype=torch.long),
            "v2_osm": osm_tag,  # NEW: Same location, so same OSM tag
            "loc_idx": self.locations.index(loc) # Slow, but simple for now. Ideally pre-compute.
        }

    def _load_sequence(self, seq_files):
        frames_sensors_files = [self._get_frame_sensor(f) for f in seq_files]
        seq_data = [np.load(f) for _, _, f in frames_sensors_files]
        seq_tensor = torch.tensor(np.stack(seq_data, axis=0), dtype=torch.float32)
        
        times = [frame for frame, _, _ in frames_sensors_files]
        sensors = [0 if s == 'S2' else 1 for _, s, _ in frames_sensors_files] # Simple mapping
        
        if self.augment:
            seq_tensor = self._apply_augmentations(seq_tensor)
            
        return seq_tensor, times, sensors

    def _sample_positive_view(self, loc):
        frames_sensors_files = self.location_map[loc]
        
        # Randomly sample a sequence of length sequence_length
        if len(frames_sensors_files) >= self.sequence_length:
            seq_files_tuple = random.sample(frames_sensors_files, self.sequence_length)
            # Sort by time to be consistent
            seq_files_tuple.sort(key=lambda x: x[0])
            seq_files = [f for _, _, f in seq_files_tuple]
        else:
            # If not enough frames, duplicate (should not happen if init is correct)
            seq_files = [f for _, _, f in frames_sensors_files] * self.sequence_length
            seq_files = seq_files[:self.sequence_length]
            
        return self._load_sequence(seq_files)

    def _get_frame_sensor(self, filepath):
        name = Path(filepath).stem
        parts = name.split('_')
        # Expecting: sensor_frame_i_j
        # But sometimes might vary.
        if len(parts) >= 4:
            sensor = parts[0]
            frame = int(parts[1])
            return frame, sensor, filepath
        else:
            # Fallback
            return 0, 'unknown', filepath
    
    def _get_osm_tag(self, loc):
        """
        Get OSM land use tag for a location from cached OSM data.
        
        Args:
            loc: Location tuple (i, j) or string
        
        Returns:
            int: OSM tag (0=unknown, 1=forest, 2=farmland, etc.)
        """
        return self.osm_cache.get(loc, 0)

    def _apply_augmentations(self, tensor):
        # Random flip
        if random.random() > 0.5:
            tensor = torch.flip(tensor, dims=[-1])
        if random.random() > 0.5:
            tensor = torch.flip(tensor, dims=[-2])
        # Random rotation 90,180,270
        k = random.randint(0,3)
        tensor = torch.rot90(tensor, k=k, dims=(-2,-1))
        # Random band jitter
        tensor = tensor + torch.randn_like(tensor) * 0.01
        return tensor
