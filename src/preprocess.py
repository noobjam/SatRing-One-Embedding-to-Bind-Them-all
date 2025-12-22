import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import rasterio
from tqdm import tqdm

from config import config
from utils import finders, raster_helpers


class BaseProductHandler(ABC):
    def __init__(self, raw_data_path: Path, processed_data_path: Path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

    @abstractmethod
    def find_products(self) -> List[Path]:
        pass

    @abstractmethod
    def group_products(self, products: List[Path]) -> Dict:
        pass

    @abstractmethod
    def get_output_dir(self, date: str, *args) -> Path:
        pass

    @abstractmethod
    def process_product(self, product: Path, date: str) -> tuple:
        pass

    def run(self):
        products = self.find_products()
        grouped = self.group_products(products)

        for date, items in tqdm(
            grouped.items(), desc=f"Processing {self.__class__.__name__}"
        ):
            for item in items:
                table, output_dir, filename = self.process_product_item(date, item)
                os.makedirs(output_dir, exist_ok=True)
                table.write_parquet(output_dir / filename)

    @abstractmethod
    def process_product_item(self, date: str, item: tuple) -> Tuple:
        pass


class S2Handler(BaseProductHandler):
    def find_products(self) -> List[Path]:
        return finders._find_s2()

    def group_products(self, products: List[Path]) -> Dict:
        return finders._group_by_date_and_tile_id(products)

    def get_output_dir(self, date: str, tile_id: str) -> Path:
        date_dir = self.processed_data_path / "S2" / date
        return date_dir / tile_id

    def process_product_item(self, date: str, item: tuple) -> Tuple:
        tile_id, product = item
        band_files = finders._find_band_file(product)
        table = raster_helpers.rasterchef(band_files, ref_band="B04", date=date)
        output_dir = self.get_output_dir(date, tile_id)
        filename = f"{product.name}.parquet"
        return table, output_dir, filename

    def process_product(self, product: Path, date: str) -> tuple:
        pass


class LSHandler(BaseProductHandler):
    def find_products(self) -> List[Path]:
        return finders._find_ls()

    def group_products(self, products: List[Path]) -> Dict:
        return finders._group_by_date_and_row_col(products)

    def get_output_dir(self, date: str, row: str, col: str) -> Path:
        date_dir = self.processed_data_path / "LS" / date
        return date_dir / row / col

    def process_product_item(self, date: str, item: tuple) -> Tuple:
        row, col, product = item
        band_files = finders._find_band_file(product)
        table = raster_helpers.rasterchef(band_files, ref_band="B04", date=date)
        output_dir = self.get_output_dir(date, row, col)
        filename = f"{product.name}.parquet"
        return table, output_dir, filename

    def process_product(self, product: Path, date: str) -> tuple:
        pass


class S1Handler(BaseProductHandler):
    def find_products(self) -> List[Path]:
        return finders._find_s1()

    def group_products(self, products: List[Path]) -> Dict:
        grouped = {}
        for product in products:
            parts = product.name.split("_")
            datetime_token = parts[4]
            date = datetime_token[:8]

            if date not in grouped:
                grouped[date] = []

            grouped[date].append((product,))

        return grouped

    def get_output_dir(self, date: str, product_name: str) -> Path:
        date_dir = self.processed_data_path / "S1" / date
        return date_dir / product_name

    def process_product_item(self, date: str, item: tuple) -> Tuple:
        product = item[0]

        measurements_dir = product / "measurement"
        if not measurements_dir.exists():
            measurements_dir = product

        pol_files = {}
        for tiff_file in measurements_dir.glob("*.tiff"):
            filename = tiff_file.stem.lower()
            if "vv" in filename:
                pol_files["VV"] = tiff_file
            elif "vh" in filename:
                pol_files["VH"] = tiff_file

        band_files = [pol_files.get("VV"), pol_files.get("VH")]
        band_files = [f for f in band_files if f is not None]

        table = raster_helpers.rasterchef(
            band_files,
            ref_band=(
                band_files[0].name.split("/")[-1].split(".")[0] if band_files else "VV"
            ),
            date=date,
        )
        output_dir = self.get_output_dir(date, product.name)
        filename = f"{product.name}.parquet"
        return table, output_dir, filename

    def process_product(self, product: Path, date: str) -> tuple:
        pass


class FusionHandler(BaseProductHandler):
    def __init__(
        self, raw_data_path: Path, processed_data_path: Path, fusion_input_path: Path
    ):
        super().__init__(raw_data_path, processed_data_path)
        self.fusion_input_path = fusion_input_path

    def find_products(self) -> List[Path]:
        products = []
        for parquet_file in self.fusion_input_path.rglob("*.parquet"):
            products.append(parquet_file)
        return products

    def group_products(self, products: List[Path]) -> Dict:
        grouped = {}
        for product in products:
            parent_dir = product.parent.name

            if parent_dir not in grouped:
                grouped[parent_dir] = []

            grouped[parent_dir].append((product,))

        return grouped

    def get_output_dir(self, date: str, product_name: str) -> Path:
        return self.processed_data_path / "fusion" / date / product_name

    def process_product_item(self, date: str, item: tuple) -> Tuple:
        parquet_file = item[0]

        table = pl.read_parquet(parquet_file)

        product_name = parquet_file.stem
        output_dir = self.get_output_dir(date, product_name)
        filename = f"{product_name}.parquet"

        return table, output_dir, filename

    def process_product(self, product: Path, date: str) -> tuple:
        pass


class Preprocess:
    def __init__(self):
        self.raw_data_path = Path(config.RAW_DATA)
        self.processed_data_path = Path(config.PROCESSED_DATA)

        self.handlers = {
            "s2": S2Handler(self.raw_data_path, self.processed_data_path),
            "ls": LSHandler(self.raw_data_path, self.processed_data_path),
            "s1": S1Handler(self.raw_data_path, self.processed_data_path),
        }

    def add_fusion_handler(self, fusion_input_path: Path):
        self.handlers["fusion"] = FusionHandler(
            self.raw_data_path, self.processed_data_path, fusion_input_path
        )

    def run(self, sensor_types: List[str] = None):
        if sensor_types is None:
            sensor_types = ["s2", "ls"]

        for sensor_type in sensor_types:
            if sensor_type in self.handlers:
                print(f"\nProcessing {sensor_type.upper()}...")
                self.handlers[sensor_type].run()
            else:
                print(f"Warning: No handler found for {sensor_type}")
