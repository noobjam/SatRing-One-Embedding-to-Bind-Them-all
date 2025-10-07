import io
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class Dataset(Dataset):
    def __init__(self, processed_data, sensor_type: str = None):
        self.processed_data = Path(processed_data)
        self.sensor_type = sensor_type if sensor_type else ["S2", "S1", "L8_9"]
        self.invetory = [(self.sensor_type, p) for p in (self.processed_data / self.sensor_type).rglob('*.npy')]
        print(f"Found {len(self.invetory)} files for sensor type {self.sensor_type}")

    def create_pactches(self):
        if self.sensor_type == 'S2':
            patch_size, stride = 128, 64
            for frame in range(len(self)):
                data, file_name = self[frame]
                out_dir = Path(self.processed_data) / self.sensor_type / file_name.replace('.npy', '') / "patches"
                os.makedirs(out_dir, exist_ok=True)
                H, W, C = data.shape
                for i in range(0, H - patch_size + 1, stride):
                    for j in range(0, W - patch_size + 1, stride):
                        patch = data[i:i + patch_size, j:j + patch_size, :]
                        np.save(out_dir / f"patch_{frame}_{i}_{j}.npy", patch)

        if self.sensor_type == 'S1':
            pass

        elif self.sensor_type == 'L8_9':
            patch_size, stride = 128, 64
            for frames in range(len(self)):
                data, file_name = self[frames]
                out_dir = Path(self.processed_data) / self.sensor_type / file_name.replace('.npy', '') / "patches"
                os.makedirs(out_dir, exist_ok=True)
                print(f"Creating patches for {file_name}...")
                H, W, C = data.shape
                for i in range(0, H - patch_size + 1, stride):
                    for j in range(0, W - patch_size + 1, stride):
                        patch = data[i:i + patch_size, j:j + patch_size, :]
                        np.save(out_dir / f"patch_{frames}_{i}_{j}.npy", patch)

        elif self.sensor_type == 'fusion':
            patch_size, stride = 224, 112
            for frames in range(len(self)):
                data, file_name = self[frames]
                out_dir = Path(self.processed_data) / self.sensor_type / file_name.replace('.npy', '') / "patches"
                os.makedirs(out_dir, exist_ok=True)
                H, W, C = data.shape
                for i in range(0, H - patch_size + 1, stride):
                    for j in range(0, W - patch_size + 1, stride):
                        patch = data[i:i + patch_size, j:j + patch_size, :]
                        np.save(out_dir / f"patch_{frames}_{i}_{j}.npy", patch)

    def get_patches(self):
        pass

    def __len__(self):
        return len(self.invetory)

    def __getitem__(self, idx):
        sensor_type, file_path = self.invetory[idx]
        data = np.load(file_path)
        data = torch.tensor(data, dtype=torch.float32)
        return data, file_path.name
