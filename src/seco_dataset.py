import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import config

CLOUD_BINS = [0, 0.10, 0.50, 1.0]
CLOUD_LABELS = ["clear", "cloudy", "rejected"]
MIN_TEMPORAL_VIEWS = 2
MAX_CLOUD_COVER = 0.10


class SeCoDataset(Dataset):
    """
    SeCo Dataset for Multi-Subspace Contrastive Learning.

    This dataset implements the specific logic required for SeCo (Seasonal Contrast) training and
    Foundation Model pre-training using temporal sequences.

    Key Features:
    - strict Cloud Quality Gating (OMN band only).
    - Multi-Subspace metadata generation (Time, Sensor, OSM).
    - Support for both Snapshot Pairs and Temporal Sequence Pairs.
    - Variable Sequence Length support with efficient Padding/Masking.

    Args:
        tiles_dir (str): Root directory containing tile parquet folders.
        patches_csv (str): Path to CSV containing patch metadata.
        patch_size (int): Spatial size of the patch (default: 224).
        patch_size_m (int): Spatial coverage in meters (default: 2240).
        sequence_length (int): Number of frames to sample.
                               1 = Snapshots.
                               >1 = Fixed Sequences.
                               0 = Full Variable Sequences (all clear frames).
        max_sequence_length (int): Buffer size for padding variable sequences.
        mode (str): Dataset mode (default 'seco').
        augment (bool): Whether to apply spatial augmentations.
        sensor_types (list): List of allowed sensors (default ['S2']).
    """

    def __init__(
        self,
        patches_dir: str,
        patch_size: int = 224,
        sequence_length: int = 1,  # >1 enables sequence mode. 0 = Full available.
        max_sequence_length: int = 60,  # Buffer size for padding if variable/full
        mode: str = "seco",
        augment: bool = True,
        sensor_types: list = None,
    ):
        self.patches_dir = Path(patches_dir)
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        self.max_sequence_length = max_sequence_length
        self.mode = mode
        self.augment = augment
        self.sensor_types = sensor_types if sensor_types else ["S2"]

        self.valid_patches = []
        self._index_dataset()

    def _index_dataset(self):
        """
        Indexes pre-filtered .npy patches from the binned directory structure.
        Structure: patches_dir / bin_name / tile_name / patch_{id}_{sensor}_{date}.npy
        """
        print(f"Indexing patches from {self.patches_dir}...")

        # Mapping: (tile, patch_id) -> list of {bin, sensor, date, file}
        patch_groups = defaultdict(list)

        # Walk through the binned directory
        for bin_dir in self.patches_dir.iterdir():
            if not bin_dir.is_dir():
                continue

            bin_name = bin_dir.name

            for tile_dir in bin_dir.iterdir():
                if not tile_dir.is_dir():
                    continue

                tile_id = tile_dir.name

                for patch_file in tile_dir.glob("patch_*.npy"):
                    # Filename format: patch_{id}_{sensor}_{date}.npy
                    # Example: patch_00246_LS_20240430.npy
                    parts = patch_file.stem.split("_")
                    if len(parts) < 4:
                        continue

                    patch_id = parts[1]
                    sensor_str = parts[2]  # LS or S2
                    date_str = parts[3]

                    patch_groups[(tile_id, patch_id)].append(
                        {
                            "bin": bin_name,
                            "sensor": 1 if sensor_str == "LS" else 0,
                            "date": date_str,
                            "file": str(patch_file),
                            "tile": tile_id,
                        }
                    )

        self.valid_patches = []
        for (tile_id, patch_id), views in patch_groups.items():
            if len(views) >= MIN_TEMPORAL_VIEWS:
                self.valid_patches.append(
                    {"tile": tile_id, "patch_id": patch_id, "frames": views}
                )

        print(
            f"Retained {len(self.valid_patches)} patches with >= {MIN_TEMPORAL_VIEWS} views."
        )

    def _extract_date(self, path):
        """
        Extracts acquisition date from filename.
        Format: patch_{id}_{sensor}_{date}.npy
        """
        parts = Path(path).stem.split("_")
        if len(parts) >= 4:
            return parts[3]
        return None

    def __len__(self):
        return len(self.valid_patches)

    def __getitem__(self, idx):
        item = self.valid_patches[idx]
        frames = item["frames"]

        # SeCo Pair Sampling:
        # Returns two sequences metadata lists
        seq1_meta, seq2_meta = self._sample_sequence_pair(frames)

        # 1. Load Data (Valid frames only) [T, C, H, W]
        raw_img1 = self._load_raw_sequence(seq1_meta)
        raw_img2 = self._load_raw_sequence(seq2_meta)

        # 2. Apply Augmentations (Spatial Invariance) - on valid frames only
        aug_img1 = self._apply_augmentations(raw_img1)
        aug_img2 = self._apply_augmentations(raw_img2)

        # 3. Handle Padding / Sequence Length
        if self.sequence_length <= 0:
            buffer_len = self.max_sequence_length
        else:
            buffer_len = self.sequence_length

        img1, mask1 = self._pad_sequence_tensor(aug_img1, buffer_len)
        img2, mask2 = self._pad_sequence_tensor(aug_img2, buffer_len)

        # Metadata
        doy1 = self._pad_meta(
            [self._get_doy(f["date"]) for f in seq1_meta], buffer_len, pad_val=0
        )
        doy2 = self._pad_meta(
            [self._get_doy(f["date"]) for f in seq2_meta], buffer_len, pad_val=0
        )

        # Sensor IDs (stored in metadata during indexing)
        s1 = self._pad_meta([f["sensor"] for f in seq1_meta], buffer_len, pad_val=0)
        s2 = self._pad_meta([f["sensor"] for f in seq2_meta], buffer_len, pad_val=0)

        return {
            "view1": {
                "image": img1,
                "mask": mask1,
                "time": torch.tensor(doy1, dtype=torch.float32),
                "sensor": torch.tensor(s1, dtype=torch.long),
            },
            "view2": {
                "image": img2,
                "mask": mask2,
                "time": torch.tensor(doy2, dtype=torch.float32),
                "sensor": torch.tensor(s2, dtype=torch.long),
            },
            "tile": item["tile"],
            "patch_id": item["patch_id"],
        }

    def _load_raw_sequence(self, seq_meta):
        tensors = []
        for meta in seq_meta:
            t = self._load_patch_tensor(meta["file"])  # [C, H, W]
            tensors.append(t)

        if not tensors:
            return torch.zeros((0, 7, self.patch_size, self.patch_size))

        return torch.stack(tensors)  # [T, C, H, W]

    def _pad_sequence_tensor(self, seq_tensor, buffer_len):
        # seq_tensor: [ActualT, C, H, W]
        actual_len = seq_tensor.shape[0]

        # Mask
        mask = torch.cat(
            [
                torch.ones(actual_len, dtype=torch.bool),
                torch.zeros(buffer_len - actual_len, dtype=torch.bool),
            ]
        )

        if actual_len >= buffer_len:
            # Crop if too long (rare due to sampling logic, but safe guard)
            return seq_tensor[:buffer_len], mask[:buffer_len]

        # Pad
        pad_len = buffer_len - actual_len
        pad_shape = (pad_len, 7, self.patch_size, self.patch_size)
        padding = torch.zeros(pad_shape, dtype=seq_tensor.dtype)

        full_tensor = torch.cat([seq_tensor, padding], dim=0)
        return full_tensor, mask

    def _sample_sequence_pair(self, frames):
        """
        Samples a pair of sequences (view1, view2) from the available frames.
        Leverages pre-filtered cloud bins for quality diversity.
        """
        # Sort frames by date first for logical sequences
        sorted_frames = sorted(frames, key=lambda x: x["date"])
        N = len(sorted_frames)

        # Separate into bins
        clean_frames = [f for f in sorted_frames if f["bin"] == "0-10"]
        cloudy_frames = [f for f in sorted_frames if f["bin"] == "10-30"]
        all_ok_frames = clean_frames + cloudy_frames

        def sample_subset(pool, target_len):
            if not pool:
                return []
            if len(pool) <= target_len:
                if self.sequence_length > 0:
                    return random.choices(pool, k=target_len)
                return pool
            return random.sample(pool, k=target_len)

        # Target lengths for SeCo
        if self.sequence_length == -1:
            # Simple random L for both views
            l1 = random.randint(1, min(N, self.max_sequence_length))
            l2 = random.randint(1, min(N, self.max_sequence_length))
        elif self.sequence_length == 0:
            l1, l2 = self.max_sequence_length, self.max_sequence_length
        else:
            l1, l2 = self.sequence_length, self.sequence_length

        # View 1 (Anchor): Always try to keep it clean (0-10)
        v1_pool = clean_frames if clean_frames else all_ok_frames
        seq1 = sample_subset(v1_pool, l1)

        # View 2 (Positive): Can include 10-30 for cloud invariance
        seq2 = sample_subset(all_ok_frames, l2)

        return sorted(seq1, key=lambda x: x["date"]), sorted(
            seq2, key=lambda x: x["date"]
        )

    def _get_doy(self, date_str):
        try:
            return pd.to_datetime(str(date_str), format="%Y%m%d").dayofyear
        except:
            return 0

    def _pad_meta(self, meta_list, length, pad_val=0):
        if len(meta_list) >= length:
            return meta_list[:length]
        return meta_list + [pad_val] * (length - len(meta_list))

    def _load_patch_tensor(self, npy_file):
        """
        Loads pre-cropped .npy patch.
        Patch array expected: [13, H, W] where last band is cloud or ignored.
        We return [7, H, W] for the model (standard multispectral).
        """
        try:
            patch = np.load(npy_file)  # [C, H, W]
            if patch.shape[0] >= 7:
                tensor = torch.from_numpy(patch[:7, :, :]).float()
            else:
                # Fallback blank
                tensor = torch.zeros((7, self.patch_size, self.patch_size))
            return tensor
        except Exception:
            return torch.zeros((7, self.patch_size, self.patch_size))

    def _df_to_tensor(self, df):
        # Kept for compatibility if needed elsewhere, but not used in .npy mode
        pass

    def _apply_augmentations(self, seq_tensor):
        # seq_tensor: [T, C, H, W]
        if not self.augment:
            return seq_tensor

        # Spatial Invariance (Z0)
        # Random Horizontal Flip
        if random.random() > 0.5:
            seq_tensor = torch.flip(seq_tensor, dims=[-1])
        # Random Vertical Flip
        if random.random() > 0.5:
            seq_tensor = torch.flip(seq_tensor, dims=[-2])
        # Random Rotation
        k = random.randint(0, 3)
        seq_tensor = torch.rot90(seq_tensor, k=k, dims=(-2, -1))

        # Gaussian Blur (simulate atmospheric effects)
        if random.random() > 0.5:
            # simple smoothing approximation or use torchvision transforms if available
            # Adding simple noise instead as defined
            pass

        # Random Gaussian Noise (Sensor Variance Z2 robustness)
        if random.random() > 0.3:
            seq_tensor = seq_tensor + torch.randn_like(seq_tensor) * 0.02

        return seq_tensor
