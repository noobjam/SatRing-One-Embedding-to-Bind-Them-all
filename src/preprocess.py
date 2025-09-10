# Assume we have S2 , S1 and L8/9 data downloaded and unzipped in the following structure:
# data/raw/
# ├── S2
# │   ├── S2A_MSIL1C_20220101T000000_N0209_R000_T00XXX_20220101T000000.SAFE
# │   ├── S2B_MSIL1C_20220101T000000_N0209_R000_T00XXX_20220101T000000.SAFE
# │   └── ...
# ├── S1
# │   ├── S1A_IW_GRDH_1SDV_20220101T000000_20220101T000000_000000_000000_000000.zip
# │   ├── S1B_IW_GRDH_1SDV_20220101T000000_20220101T000000_000000_000000_000000.zip
# │   └── ...
# └── L8_9
#     ├── LC08_L1TP_000000_20220101_20220101_01_T1.tar.gz
#     ├── LC09_L1TP_000000_20220101_20220101_01_T1.tar.gz
#     └── ...       



#TODO: before stacking, for S2 bands(~SCL)/=10000 

import glob
import os
from pathlib import Path
from typing import List, Sequence
# import rasterio
import numpy as np
import rasterio


class Preprocess:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = processed_data_path

    def _discover_products(self, sensor_type:str):
        #Discover files based on sensor type (S2, S1, L8/9)
        products: List[Path] = []
        for child in self.raw_data_path.iterdir():
            if child.is_dir() and child.name.startswith(sensor_type):
                for safe_file in child.iterdir():
                    if safe_file.suffix == '.SAFE' and safe_file.is_dir():
                        products.append(safe_file)

                
        products.sort()
        return products



    
    def _find_band_file(self,product, subdir_filter:str = 'IMG_DATA/', extensions: Sequence = ('.tif', '.jp2', '.tiff')):

        files = []

        for p in product.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in [ext.lower() for ext in extensions]:
                continue
            if subdir_filter and subdir_filter not in str(p):
                continue
            
            files.append(p)
            
        return files
    
    def _resample_band(self, band_path: Path, reference_band_path: Path, resampling_method: str = 'cubic'):
    
        with rasterio.open(reference_band_path) as ref_src:
            ref_height, ref_width = ref_src.height, ref_src.width
        
        with rasterio.open(band_path) as src:
            # Simple resize to reference dimensions( cuz _get_metadata shows :
            """:{'crs': None, 'transform': Affine(1.0, 0.0, 0.0,
                0.0, 1.0, 0.0), 'width': 488, 'height': 486, 'dtype': 'uint16', 'count': 1} )"""
            data = src.read(
                out_shape=(src.count, ref_height, ref_width),
                resampling=getattr(rasterio.enums.Resampling, resampling_method)
            )
            
            return data, None, None, src.dtypes[0]
            

    def _get_metadata(self, band_path: Path):

        with rasterio.open(band_path) as src:
            return {
                'crs': src.crs,
                'transform': src.transform,
                'width': src.width,
                'height': src.height,
                'dtype': src.dtypes[0],
                'count': src.count
            }


    def preprocess_s2(self):
        target_bands = {'10m': ['B02','B03','B04','B08'],
                 '20m': ['B01','B05','B06','B07','B8A','B11','B12','SCL'],
                 '60m': ['B09']}
        # product-type: S2MSL2A
        # resample , stack: [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12,SCL] ---> (H,W,13)
        for product in self._discover_products('S2'):
            bands_10m= self._find_band_file(product, subdir_filter='IMG_DATA/R10m/')
            bands_20m= self._find_band_file(product, subdir_filter='IMG_DATA/R20m/')
            bands_60m= self._find_band_file(product, subdir_filter='IMG_DATA/R60m/')


            #group target bands starting from 10m, if same band found in multiple resolutions, prefer higher resolution
            selected_bands = []
            for band in target_bands['10m']:
                for bfile in bands_10m:
                    if band in bfile.name:
                        selected_bands.append(bfile)
                        
            for band in target_bands['20m']:
                for bfile in bands_20m:
                    if band in bfile.name:
                        selected_bands.append(bfile)
            for band in target_bands['60m']:
                for bfile in bands_60m:
                    if band in bfile.name:
                        selected_bands.append(bfile)
            sorted_bands = sorted(selected_bands, key=lambda x: x.name)
            selected_bands = sorted_bands
            print(f"Selected bands for {product.name}: {[b.name for b in selected_bands]}")
            # resample all bands to 10 m (normal bands with bicubic, SCL with nearest) using rasterio
            # stack bands into a single array (H,W,14)


            for i, band in enumerate(selected_bands):
                if 'SCL' in band.name:
                    resampling_method = 'nearest'
                else:
                    resampling_method = 'cubic'

                reference_band=bands_10m[0]
                data, _, _, _ = self._resample_band(band, reference_band_path=reference_band, resampling_method=resampling_method)
                print(f"Resampled {band.name} to 10m with shape {data.shape}")
                if i == 0:
                    stacked_data = data # (1,H,W)
                else: 
                    stacked_data = np.vstack((stacked_data, data)) # (N,H,W)
            #transpose to (H,W,C)
            stacked_data = np.transpose(stacked_data, (1, 2, 0))  # (H,W,C)
            print(f"Processed {product.name}, stacked shape: {stacked_data.shape}")
            # Save the stacked array as a .npy file (maybe change this to tif, Hamza'z input required, npy easier for torch's from_numpy but no geo metadata)
            output_file = Path(self.processed_data_path) / f"{product.name}_stacked.npy"
            np.save(output_file, stacked_data)


                
    def preprocess_s1(self):
        # product-type: GRD
        # resample , stack: [VV,VH] ---> (H,W,2)
        for file in self._discover_files('S1'):
            print(f"Processing {file}...")
        pass

    def preprocess_l8_9(self):
        # product-type: XL2SR 
        #resample , stack: [SR_B1,SR_B2,SR_B3,SR_B4,SR_B5,SR_B6,SR_B7,QA_PIXEL] ---> (H,W,8)
        for file in self._discover_files('L8_9'):
            print(f"Processing {file}...")

        pass

    def run(self):
        self.preprocess_s2()
        self.preprocess_s1()
        self.preprocess_l8_9()