import io
import os
import torch
import zipfile
import requests
import torch.nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


class Dataset(Dataset):
    def __init__(self, processed_data, sensor_type: str = None):
        self.processed_data = Path(processed_data)
        self.sensor_type = sensor_type if sensor_type else ["S2", "S1", "L8_9"]
        self.invetory =  [( self.sensor_type, p ) for p in (self.processed_data / self.sensor_type).rglob('*.npy')]
        print(f"Found {len(self.invetory)} files for sensor type {self.sensor_type}")


    def create_pactches(self):
        if self.sensor_type == 'S2':
            # S2 is (H,W,13)
            patch_size = 128
            stride = 64
            for frame in range(len(self)):
                data,file_name = self[frame]
                os.makedirs(Path(self.processed_data) /self.sensor_type/ file_name.replace('.npy','')/ "patches" , exist_ok=True)


                # assuming data is a numpy array of shape (H, W, C)
                H, W, C = data.shape
                for i in range(0, H - patch_size + 1, stride):
                    for j in range(0, W - patch_size + 1, stride):
                        patch = data[i:i + patch_size, j:j + patch_size, :]
                        #save patch as .npy file
                        patch_file = Path(self.processed_data) /self.sensor_type/ file_name.replace('.npy','')/ "patches" / f"patch_{frame}_{i}_{j}.npy"
                        np.save(patch_file, patch)

        if self.sensor_type == 'S1':
            # S1 is (H,W,2)
            pass
     
                        

        elif self.sensor_type == 'L8_9':
            # L8_9 is (H,W,11)
            patch_size = 128
            stride = 64
            for frames in range(len(self)):
                data,file_name = self[frames]

                os.makedirs(Path(self.processed_data) /self.sensor_type / file_name.replace('.npy','') / "patches" , exist_ok=True)
                print(f"Creating patches for {file_name}...")

                H, W, C = data.shape
                for i in range(0, H - patch_size + 1, stride):
                    for j in range(0, W - patch_size + 1, stride):
                        patch = data[i:i + patch_size, j:j + patch_size, :]
                        #save patch as .npy file
                        patch_file = Path(self.processed_data)/self.sensor_type/file_name.replace('.npy','') / "patches" / f"patch_{frames}_{i}_{j}.npy"
                        np.save(patch_file, patch)




    def __len__(self):
        return len(self.invetory)       
    def __getitem__(self, idx):
        sensor_type, file_path = self.invetory[idx]
        data = np.load(file_path)
        data = torch.tensor(data, dtype=torch.float32)
        return data,file_path.name



