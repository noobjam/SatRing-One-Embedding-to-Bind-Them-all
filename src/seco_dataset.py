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
from tqdm import tqdm

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
        tiles_dir: str,
        patches_csv: str,
        patch_size: int = 224,
        patch_size_m: int = 2240,
        sequence_length: int = 1,  # >1 enables sequence mode. 0 = Full available.
        max_sequence_length: int = 60,  # Buffer size for padding if variable/full
        mode: str = "seco",
        augment: bool = True,
        sensor_types: list = None,
    ):
        self.tiles_dir = Path(tiles_dir)
        self.patch_size = patch_size
        self.patch_size_m = patch_size_m
        self.sequence_length = sequence_length
        self.max_sequence_length = max_sequence_length
        self.mode = mode
        self.augment = augment
        self.sensor_types = sensor_types if sensor_types else ["S2"]

        # OMN values: 0=Clear. STRICT GATING logic relies on this.
        self.valid_omn_values = [0]

        print(f"Loading patches from {patches_csv}...")
        self.patches_df = pd.read_csv(patches_csv)

        self.valid_patches = []
        self._index_dataset()

    def _index_dataset(self):
        print(
            f"Indexing patches and checking cloud quality (Threshold: <{MAX_CLOUD_COVER*100}% cloud)..."
        )
        valid_patches = []

        # Pre-scan tiles_dir for all parquets to avoid repeated recursive searches
        all_parquets = list(self.tiles_dir.rglob("*.parquet"))
        print(f"Found {len(all_parquets)} Parquet files total.")

        # We process row by row
        for idx, row in tqdm(
            self.patches_df.iterrows(), total=len(self.patches_df), desc="Indexing"
        ):
            tile_id = row["tile"]

            # Find all parquets matching this tile_id in the filename
            # Format: LS8_..._T35MPT_...parquet or S2A_..._T35MPT_...parquet
            tile_parquets = [p for p in all_parquets if f"_{tile_id}_" in p.name]

            if not tile_parquets:
                continue

            # Load temporal availability
            available_frames = self._find_available_frames_from_list(tile_parquets, row)

            if len(available_frames) < MIN_TEMPORAL_VIEWS:
                continue

            valid_patches.append(
                {
                    "patch_id": idx,
                    "patch_info": row.to_dict(),
                    "frames": available_frames,
                }
            )

        self.valid_patches = valid_patches
        print(
            f"Retained {len(self.valid_patches)} / {len(self.patches_df)} patches with >= {MIN_TEMPORAL_VIEWS} clear views."
        )

    def _find_available_frames_from_list(self, parquet_files, patch_info):
        """
        Processes a list of parquet files and checks cloud cover for the specific patch.
        Matches OMN band specifically.
        """
        valid_frames = []
        bbox = self._compute_bbox(patch_info["lon"], patch_info["lat"])

        for pq_file in sorted(parquet_files):
            try:
                date_str = self._extract_date(pq_file)
                if not date_str:
                    continue

                # Identify Sensor
                # LS8/LS9 -> 1, S2A/S2B -> 0
                sensor_name = pq_file.name[:3].upper()
                sensor_id = 1 if sensor_name.startswith("LS") else 0

                df = pl.read_parquet(pq_file)

                # Column check
                cols = df.columns
                x_col = "x" if "x" in cols else "longitude"
                y_col = "y" if "y" in cols else "latitude"

                patch_df = df.filter(
                    (pl.col(x_col) >= bbox[0])
                    & (pl.col(x_col) <= bbox[2])
                    & (pl.col(y_col) >= bbox[1])
                    & (pl.col(y_col) <= bbox[3])
                )

                if len(patch_df) < (self.patch_size // 10) ** 2:
                    continue

                # Cloud Check (OMN column strictly)
                cloud_cover = self._compute_cloud_cover(patch_df)

                if cloud_cover <= MAX_CLOUD_COVER:
                    valid_frames.append(
                        {
                            "date": date_str,
                            "file": str(pq_file),
                            "cloud_cover": cloud_cover,
                            "sensor_id": sensor_id,
                        }
                    )

            except Exception:
                continue

        return valid_frames

    def _find_available_frames(self, tile_path, patch_info):
        """
        Scans the tile directory for parquet files and checks cloud cover for the specific patch.
        This is expensive if done for every patch fully, so we assume 'resampled' parquet structure
        allows efficient bounding box filtering.
        """
        valid_frames = []

        # Look for parquet files recursively
        parquet_files = sorted(list(tile_path.rglob("*.parquet")))

        bbox = self._compute_bbox(patch_info["lon"], patch_info["lat"])

        for pq_file in parquet_files:
            try:
                # Extract date from filename or parent folder
                date_str = self._extract_date(pq_file)
                if not date_str:
                    continue

                # Lazy load header/schema check if possible, but for now try to read specific region
                # Assuming spatial partitioning or optimal layout

                # Helper to read just the cloud information first if separated?
                # For now, we read the patch
                df = pl.read_parquet(pq_file)

                # Check column existence check
                if "x" in df.columns:
                    x_col, y_col = "x", "y"
                else:
                    x_col, y_col = "longitude", "latitude"

                patch_df = df.filter(
                    (pl.col(x_col) >= bbox[0])
                    & (pl.col(x_col) <= bbox[2])
                    & (pl.col(y_col) >= bbox[1])
                    & (pl.col(y_col) <= bbox[3])
                )

                if (
                    len(patch_df) < (self.patch_size // 10) ** 2
                ):  # Rough check for completeness
                    continue

                # Cloud Check
                cloud_cover = self._compute_cloud_cover(patch_df)

                if cloud_cover <= MAX_CLOUD_COVER:
                    valid_frames.append(
                        {
                            "date": date_str,
                            "file": str(pq_file),
                            "cloud_cover": cloud_cover,
                        }
                    )

            except Exception:
                continue

        return valid_frames

    def _extract_date(self, path):
        """
        Strictly extracts acquisition date from Fusion Product filename.
        Format: LS8_OLIL2F_20240117T081412_... -> returns 20240117
        """
        import re

        stem = Path(path).stem
        # Regex for the first timestamp after sensor/product tokens
        # Typically the 3rd element in underscore-separated name
        match = re.search(r"_(\d{8})T", stem)
        if match:
            return match.group(1)

        # Fallback to pure digit check if first fails
        match = re.search(r"(\d{8})", stem)
        if match:
            return match.group(1)
        return None

    def _compute_bbox(self, cx, cy):
        half = self.patch_size_m / 2
        return (cx - half, cy - half, cx + half, cy + half)

    def _compute_cloud_cover(self, df):
        if "OMN" not in df.columns:
            return 1.0  # Assume cloudy if no mask

        # OMN: 0=Clear. Everything else is cloud/shadow for us.
        omn = df["OMN"].to_numpy()
        non_clear = (omn != 0).sum()
        return non_clear / len(omn)

    def __len__(self):
        return len(self.valid_patches)

    # Valid OSM Land Use Labels (Mock/Example mapping)
    # 0: Unknown, 1: Forest, 2: Cropland, 3: Grassland, 4: Water, 5: Built-up, 6: Wetland

    def __getitem__(self, idx):
        item = self.valid_patches[idx]
        frames = item["frames"]
        patch_info = item["patch_info"]

        # SeCo Pair Sampling:
        # Returns two sequences metadata lists
        seq1_meta, seq2_meta = self._sample_sequence_pair(frames)

        # 1. Load Data (Valid frames only) [T, C, H, W]
        raw_img1 = self._load_raw_sequence(seq1_meta, patch_info)
        raw_img2 = self._load_raw_sequence(seq2_meta, patch_info)

        # 2. Apply Augmentations (Spatial Invariance) - on valid frames only
        # This keeps padding pure zeros
        aug_img1 = self._apply_augmentations(raw_img1)
        aug_img2 = self._apply_augmentations(raw_img2)

        # 3. Handle Padding / Sequence Length
        # Determine buffer length.
        # If Fixed > 0, buffer is sequence_length.
        # If Variable (0 or -1), buffer is max_sequence_length.
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

        # Sensor
        s1 = self._pad_meta([f["sensor_id"] for f in seq1_meta], buffer_len, pad_val=0)
        s2 = self._pad_meta([f["sensor_id"] for f in seq2_meta], buffer_len, pad_val=0)

        # Semantic
        label_map = {
            "Forest": 1,
            "Cropland": 2,
            "Grassland": 3,
            "Water": 4,
            "Built_Up_Area": 5,
            "Wetland": 6,
            "Other": 0,
        }
        dom_class = patch_info.get("dominant_class", "Other")
        osm_label = label_map.get(dom_class, 0)

        return {
            "view1": {
                "image": img1,
                "mask": mask1,
                "time": torch.tensor(doy1, dtype=torch.float32),
                "sensor": torch.tensor(s1, dtype=torch.long),
                "osm": torch.tensor(osm_label, dtype=torch.long),
            },
            "view2": {
                "image": img2,
                "mask": mask2,
                "time": torch.tensor(doy2, dtype=torch.float32),
                "sensor": torch.tensor(s2, dtype=torch.long),
                "osm": torch.tensor(osm_label, dtype=torch.long),
            },
            "tile": patch_info["tile"],
            "lon": patch_info["lon"],
            "lat": patch_info["lat"],
        }

    def _load_raw_sequence(self, seq_meta, patch_info):
        tensors = []
        for meta in seq_meta:
            t = self._load_patch_tensor(meta["file"], patch_info)  # [C, H, W]
            tensors.append(t)

        if not tensors:
            # Should guard against empty but assume at least 1 exists
            return torch.zeros((0, 12, self.patch_size, self.patch_size))

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
        pad_shape = (pad_len, 12, self.patch_size, self.patch_size)
        padding = torch.zeros(pad_shape, dtype=seq_tensor.dtype)

        full_tensor = torch.cat([seq_tensor, padding], dim=0)
        return full_tensor, mask

    def _sample_sequence_pair(self, frames):
        """
        Samples a pair of sequences (view1, view2) from the available frames.
        If sequence_length == -1, it implements a mixed strategy to cover:
        - Snapshots (L=1, L=1)
        - Full Sequences (L=Full, L=Full)
        - Scaled Sequences (L=k, L=k) where k is random
        - Asymmetric Sequences (L=k1, L=k2)
        """
        # Sort frames by date first for logical sequences
        sorted_frames = sorted(frames, key=lambda x: x["date"])
        N = len(sorted_frames)

        def sample_subset(pool, target_len):
            if not pool:
                return []

            # If N <= target_len, we take what we have (or repeat if fixed mode required,
            # but here padding is handled later for variable modes)
            if N <= target_len:
                if self.sequence_length > 0:  # Fixed length mode
                    return random.choices(pool, k=target_len)
                return pool
            else:
                # Random sample for variety
                return random.sample(pool, k=target_len)

        # Mode Selection
        if self.sequence_length == -1:
            # Random Mixed Mode: Roll for strategy
            strat = random.random()
            if strat < 0.2:
                # Case: Snapshots
                l1, l2 = 1, 1
            elif strat < 0.4:
                # Case: Full Sequences
                l1, l2 = self.max_sequence_length, self.max_sequence_length
            elif strat < 0.7:
                # Case: Incremental Scaled (Same length L)
                L = random.randint(1, min(N, self.max_sequence_length))
                l1, l2 = L, L
            else:
                # Case: Truly Asymmetric
                l1 = random.randint(1, min(N, self.max_sequence_length))
                l2 = random.randint(1, min(N, self.max_sequence_length))
        elif self.sequence_length == 0:
            # Full Mode
            l1, l2 = self.max_sequence_length, self.max_sequence_length
        else:
            # Fixed Mode
            l1, l2 = self.sequence_length, self.sequence_length

        # Sample Sequences
        seq1 = sample_subset(sorted_frames, l1)

        # For view2, we re-sample to get temporal diversity
        # but if N is small or it's a "Full" request, overlaps will occur.
        seq2 = sample_subset(sorted_frames, l2)

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

    def _load_patch_tensor(self, pq_file, patch_info):
        bbox = self._compute_bbox(patch_info["lon"], patch_info["lat"])
        try:
            df = pl.read_parquet(pq_file)

            # Smart column detection
            cols = df.columns
            x_col = "x" if "x" in cols else "longitude"
            y_col = "y" if "y" in cols else "latitude"

            patch_df = df.filter(
                (pl.col(x_col) >= bbox[0])
                & (pl.col(x_col) <= bbox[2])
                & (pl.col(y_col) >= bbox[1])
                & (pl.col(y_col) <= bbox[3])
            )
            return self._df_to_tensor(patch_df)
        except Exception:
            return torch.zeros((12, self.patch_size, self.patch_size))

    def _df_to_tensor(self, df):
        bands = [
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B9",
            "B11",
            "B12",
        ]
        tensor_list = []

        # Check if empty
        if len(df) == 0:
            return torch.zeros((12, self.patch_size, self.patch_size))

        for b in bands:
            if b in df.columns:
                data = df[b].to_numpy().astype(np.float32)
                if len(data) == self.patch_size * self.patch_size:
                    img = data.reshape(self.patch_size, self.patch_size)
                else:
                    # Robust resize/fill
                    img = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                    side = int(np.sqrt(len(data)))
                    if side**2 == len(data):
                        img[:side, :side] = data.reshape(side, side)
            else:
                img = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            tensor_list.append(torch.tensor(img))

        return torch.stack(tensor_list)

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
