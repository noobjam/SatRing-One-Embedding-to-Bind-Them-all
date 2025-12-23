import re
from pathlib import Path


def _find_s2(path):
    base_s2 = path / "S2"

    raw_s2 = []
    for target_safe in Path(base_s2).glob("*.SAFE"):
        if target_safe.is_dir():
            raw_s2.append(Path(target_safe))

    return sorted(raw_s2)


def _find_ls(path):

    base_ls = path / "LS"
    raw_ls = []
    for target in Path(base_ls).glob("*"):
        if target.is_dir() and target.name.endswith("_T1"):
            raw_ls.append(Path(target))
    return sorted(raw_ls)


def find_bands(path, product_type):
    if product_type == "S2":
        bands = {}
        files = list(path.glob("**/IMG_DATA/R10/*.jp2"))
        # r20_files = list(path.glob("**/IMG_DATA/R20/*.jp2"))

        for f in files:
            match = re.search(r"_B(\d{1,2})_", f.name)
            if match:
                band = f"B{match.group(1)}"
                if band in [
                    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12",
                ]:
                    bands[band] = f

        # SCL: Scene Classification Layer (for cloud gating)
        scl_files = list(path.glob("**/IMG_DATA/R20/*_SCL_*.jp2"))
        if scl_files:
            bands["SCL"] = scl_files[0]

        return bands

    if product_type == "LS":
        bands = {}
        tif_files = list(path.glob("*.TIF"))
        for f in tif_files:
            # Match spectral bands SR_B1 ... SR_B11
            match = re.search(r"_SR_B(\d+)\.TIF$", f.name)
            if match:
                band = f"B{match.group(1)}"
                if band in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B10", "B11"]:
                    bands[band] = f
            
            # QA_PIXEL: Quality Assessment
            if "_QA_PIXEL" in f.name:
                bands["QA_PIXEL"] = f
                
        return bands
