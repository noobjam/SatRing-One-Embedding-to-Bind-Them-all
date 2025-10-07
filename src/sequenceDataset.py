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
        anchor, anchor_loc, anchor_sensors = self._get_anchor_sequence(idx)
        if random.random() < 0.5:
            # Positive pair
            pair = self.sample_positive_pair(anchor_loc, anchor_sensors)
            label = 1
        else:
            # Negative pair
            pair = self.sample_negative_pair(anchor_loc)
            label = 0
        return anchor, pair, label

    def _get_anchor_sequence(self, idx):
        loc, seq_files = self.sequences[idx]
        frames_sensors_files = [self._get_frame_sensor(f) for f in seq_files]
        seq_data = [np.load(f) for _, _, f in frames_sensors_files]
        seq_tensor = torch.tensor(np.stack(seq_data, axis=0), dtype=torch.float32)
        sensors = [s for _, s, _ in frames_sensors_files]
        return seq_tensor, loc, sensors

    def _get_frame_sensor(self, filepath):
        name = Path(filepath).stem
        sensor, frame, i, j = name.split('_')
        return int(frame), sensor, filepath

    def sample_positive_pair(self, loc, sensors):
        frames_sensors_files = self.location_map[loc]
        # Same location, different time
        seq = random.sample(frames_sensors_files, self.sequence_length)
        seq_data = [np.load(f) for _, _, f in seq]
        seq_tensor = torch.tensor(np.stack(seq_data, axis=0), dtype=torch.float32)
        if self.augment:
            seq_tensor = self._apply_augmentations(seq_tensor)
        return seq_tensor

    def sample_negative_pair(self, anchor_loc):
        neg_loc = random.choice([l for l in self.locations if l != anchor_loc])
        frames_sensors_files = self.location_map[neg_loc]
        seq = random.sample(frames_sensors_files, self.sequence_length)
        seq_data = [np.load(f) for _, _, f in seq]
        seq_tensor = torch.tensor(np.stack(seq_data, axis=0), dtype=torch.float32)
        return seq_tensor

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
