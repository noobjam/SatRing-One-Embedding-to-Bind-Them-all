import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import os



class PatchDataset(Dataset):
    def __init__(self,processed_dir, sensor_type):
        self.processed_dir = processed_dir
        self.sensor_type = sensor_type
        self.inventory = []

        
    def _index_pacthes(self):
        # scan processed_dir for patches
        
