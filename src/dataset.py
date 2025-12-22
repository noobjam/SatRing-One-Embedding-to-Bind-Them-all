import io
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from config import config


class Dataset(Dataset):
    def __init__(self, sensor_types: list = None, reorganize: bool = False):
        self.processed_data = Path(config.PROCESSED_DATA)
        self.sensor_types = sensor_types if sensor_types else ["S2", "LS"]

        # Define band columns for each sensor type
        self.band_columns = {
            "S2": [
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
                "B10",
                "B11",
                "B12",
            ],
            "LS": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B10", "B11"],
            # "S1": ["VV", "VH"],
        }

        self.sensor_mapping = {
            "S2": 0,
            "LS": 1,
            "S1": 2,
        }

        if reorganize:
            self.locations = self._organize_by_location()
        else:
            self.inventory = []
            for sensor_type in self.sensor_types:
                sensor_dir = self.processed_data / sensor_type
                if sensor_dir.exists():
                    # Find parquet files, excluding patches
                    for date_dir in sensor_dir.iterdir():
                        if date_dir.is_dir() and date_dir.name.isdigit():
                            for parquet_file in date_dir.glob("*.parquet"):
                                if "patches" not in str(parquet_file):
                                    self.inventory.append((sensor_type, parquet_file))
            print(
                f"Found {len(self.inventory)} files across sensor types {self.sensor_types}"
            )

    def _organize_by_location(self):
        """
        Organize data by location using unique (latitude, longitude) coordinate sets.
        Returns:
            list of dicts:
            {
                'loc_id': str,
                'timestamps': np.array([DOY...]),
                'sensors': np.array([0/1/2...]),
                'files': [paths...]
            }
        """
        location_data = defaultdict(
            lambda: {"timestamps": [], "sensors": [], "files": []}
        )

        print("Organizing data by location using latitude/longitude...")

        # Process all sensor types
        for sensor_type in self.sensor_types:
            sensor_dir = self.processed_data / sensor_type
            if not sensor_dir.exists():
                print(f"Warning: {sensor_type} directory not found")
                continue

            for date_dir in sensor_dir.iterdir():
                if not date_dir.is_dir() or not date_dir.name.isdigit():
                    continue

                date_str = date_dir.name  # YYYYMMDD

                parquet_files = [
                    p for p in date_dir.glob("*.parquet") if "patches" not in str(p)
                ]

                for file_path in parquet_files:
                    df = pl.read_parquet(file_path)

                    unique_lat = sorted(df["latitude"].unique().to_list())
                    unique_lon = sorted(df["longitude"].unique().to_list())

                    # Use a bounding-box signature as the location ID
                    loc_id = (
                        f"lat{min(unique_lat):.5f}-{max(unique_lat):.5f}_"
                        f"lon{min(unique_lon):.5f}-{max(unique_lon):.5f}"
                    )
                    # ----------------------------------------------------------

                    # Convert date to Day Of Year
                    try:
                        date_obj = datetime.strptime(date_str, "%Y%m%d")
                        doy = date_obj.timetuple().tm_yday
                    except ValueError:
                        print(f"Warning: Could not parse date from {date_str}")
                        continue

                    sensor_encoded = self.sensor_mapping[sensor_type]

                    location_data[loc_id]["timestamps"].append(doy)
                    location_data[loc_id]["sensors"].append(sensor_encoded)
                    location_data[loc_id]["files"].append(file_path)

        organized = []
        for loc_id, values in location_data.items():
            organized.append(
                {
                    "loc_id": loc_id,
                    "timestamps": np.array(values["timestamps"], dtype=np.int32),
                    "sensors": np.array(values["sensors"], dtype=np.int8),
                    "files": values["files"],
                }
            )

        print(f"Organized into {len(organized)} unique lat/lon locations.")
        return organized
