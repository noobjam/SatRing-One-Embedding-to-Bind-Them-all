import os
import glob
import argparse
import rasterio
import numpy as np
from rasterio.plot import reshape_as_image
import logging
import shutil
import tempfile
import json

def setup_logger(log_path=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()] + ([logging.FileHandler(log_path)] if log_path else [])
    )

def discover_all_bands(img_data_dir):
    band_map = {}
    for ext in ("jp2", "tif"):
        files = glob.glob(os.path.join(img_data_dir, f"*_*B??.{ext}"))
        for fpath in files:
            name = os.path.basename(fpath)
            # Pattern: ..._B02.jp2 or ..._B8A.tif
            bidx = name.split("_")[-1].split(".")[0]  
            band_map[bidx] = fpath
    return band_map

def discover_qi_bands(img_data_dir):
    qi_map = {}
    for ext in ("jp2", "tif"):
        files = glob.glob(os.path.join(img_data_dir, f"QI_*.{ext}"))
        for fpath in files:
            name = os.path.basename(fpath)
            qi_map[name] = fpath
    return qi_map

def preprocess_sentinel2_safe_all_bands(safe_dir, out_dir):
    """
    Process a Sentinel-2 .SAFE folder: preprocess all spectral bands, save proper metadata, log any missing/corrupt bands.
    """
    img_data_dirs = glob.glob(os.path.join(safe_dir, "GRANULE", "*", "IMG_DATA*"))
    if not img_data_dirs:
        logging.error(f"Could not find IMG_DATA folder in {safe_dir}")
        return False
    img_data_dir = img_data_dirs[0]

    band_map = discover_all_bands(img_data_dir)
    if not band_map:
        logging.error(f"No spectral band files found in {img_data_dir}")
        return False

    images = []
    bands = []
    meta = {}
    failed_bands = []
    for band in sorted(band_map.keys()):
        band_fp = band_map[band]
        try:
            with rasterio.open(band_fp) as src:
                arr = src.read(1).astype(np.float32)
                arr = arr / 10000.0
                images.append(arr)
                bands.append(band)
        except Exception as e:
            logging.warning(f"Failed to read band {band_fp}: {e}")
            failed_bands.append({"band": band, "file": band_fp, "error": str(e)})

    if not images:
        logging.error(f"No bands loaded for {safe_dir}")
        return False

    stacked = np.stack(images, axis=0)  # [num_bands, H, W]
    stacked = np.transpose(stacked, (1, 2, 0))  # [H, W, num_bands]

    base_name = os.path.basename(safe_dir.rstrip("/"))
    out_stack = os.path.join(out_dir, f"{base_name}_stack_allbands.npy")
    out_meta = os.path.join(out_dir, f"{base_name}_bands_meta.json")

    # Write atomically: first to temp file, then move.
    with tempfile.NamedTemporaryFile(delete=False, dir=out_dir, suffix='.npy') as tmpfile:
        np.save(tmpfile.name, stacked)
        tmpfile.flush()
        os.fsync(tmpfile.fileno())
        final_npy = out_stack
        shutil.move(tmpfile.name, final_npy)
        logging.info(f"Saved processed bands stack: {final_npy}")

    meta = {
        "bands": bands,
        "safe_dir": safe_dir,
        "band_files": [band_map[b] for b in bands],
        "failed_bands": failed_bands
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
        logging.info(f"Wrote meta: {out_meta}")

    # Copy QI bands
    qi_map = discover_qi_bands(img_data_dir)
    for qname, qpath in qi_map.items():
        try:
            shutil.copyfile(qpath, os.path.join(out_dir, f"{base_name}_{qname}"))
            logging.info(f"Copied QI band: {qname}")
        except Exception as e:
            logging.warning(f"Failed to copy QI band {qname}: {e}")

    return True

def preprocess_landsat_tiff(tiff_path, out_dir):
    """
    Process a Landsat TIFF file: normalize and save as npy (for all bands).
    """
    with rasterio.open(tiff_path) as src:
        arr = src.read().astype(np.float32)  # [bands, H, W]
        arr = arr / 10000.0
        arr = np.transpose(arr, (1, 2, 0))  # [H, W, bands]
    filename = os.path.splitext(os.path.basename(tiff_path))[0]
    out_stack = os.path.join(out_dir, f"{filename}_stack_allbands.npy")
    np.save(out_stack, arr)
    logging.info(f"Processed and saved Landsat TIFF stack: {out_stack}")
    return out_stack

def main(raw_dir="data/raw", processed_dir="data/processed"):
    os.makedirs(processed_dir, exist_ok=True)
    setup_logger(os.path.join(processed_dir, "preprocess.log"))
    for entry in os.listdir(raw_dir):
        in_path = os.path.join(raw_dir, entry)
        if entry.endswith(".SAFE") and os.path.isdir(in_path):
            logging.info(f"Processing Sentinel-2 SAFE: {in_path}")
            preprocess_sentinel2_safe_all_bands(in_path, processed_dir)
        elif entry.lower().endswith(('.tif', '.tiff')):
            logging.info(f"Processing Landsat TIFF: {in_path}")
            preprocess_landsat_tiff(in_path, processed_dir)
        # Extend here for other satellites/formats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw satellite data for ML training. Production-level, all-band.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Raw input data directory")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Output processed data directory")
    args = parser.parse_args()
    main(args.raw_dir, args.processed_dir)
