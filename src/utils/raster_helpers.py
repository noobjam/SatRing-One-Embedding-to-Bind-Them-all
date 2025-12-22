from pathlib import Path

import numpy as np
import polars as pl
import rasterio
from loguru import logger
from pyproj import CRS, Transformer
from rasterio.enums import Resampling


def rasterchef(bands, ref_band, product_type: str, date):
    # read the bands ---> resample & stack ---> add coordinates --->parquet

    # bands is a dict :
    """{
    "B1": Path("LC08_L1TP_123032_20210101_20210101_SR_B1.TIF"),
    "B2": Path("LC08_L1TP_123032_20210101_20210101_SR_B2.TIF"),
    "B3": Path("LC08_L1TP_123032_20210101_20210101_SR_B3.TIF"),
    "B4": Path("LC08_L1TP_123032_20210101_20210101_SR_B4.TIF"),
    """

    if ref_band not in bands:
        raise ValueError(f"Reference band {ref_band} not found in the list of bands.")
    ref_file = bands[ref_band]

    with rasterio.open(ref_file) as ref_src:
        ref_profile = ref_src.profile
        ref_shape = ref_src.shape
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        ref_res = ref_src.res

    band_arrays = []
    valid_band_names = []

    # --read / Resamnple

    for band_name, band_path in bands.items():
        with rasterio.open(band_path) as src:
            # Check if resampling is needed
            # SCL must be resampled using nearest neighbor,same for QA_PIXEL
            if src.shape != ref_shape or src.res[0] != ref_res[0]:
                logger.info(
                    f"Resampling {band_name} from shape {src.shape} to {ref_shape}"
                )
                data = src.read(
                    1,
                    out_shape=ref_shape,
                    resampling=(
                        Resampling.nearest
                        if (band_name in ["SCL", "QA_PIXEL"])
                        else Resampling.bilinear
                    ),
                )
            else:
                data = src.read(1)

            logger.debug(
                f"{band_name}: min={data.min()}, max={data.max()}, "
                f"mean={data.mean():.2f}, non-zero={np.count_nonzero(data)}"
            )

            band_arrays.append(data.flatten().astype(np.float32))
            valid_band_names.append(band_name)

    if not band_arrays:
        raise ValueError("No valid bands were processed")

    logger.info(f"Processed {len(valid_band_names)} bands: {valid_band_names}")

    # --stack & add coords
    # --- coordinates ---
    cols, rows = np.meshgrid(np.arange(ref_shape[1]), np.arange(ref_shape[0]))
    xs, ys = rasterio.transform.xy(ref_transform, rows, cols)
    xs = np.array(xs).flatten()
    ys = np.array(ys).flatten()

    if ref_crs.to_string() == "EPSG:4326":
        longitudes, latitudes = xs, ys
    else:
        target_crs = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(ref_crs, target_crs, always_xy=True)
        longitudes, latitudes = transformer.transform(xs, ys)

        data_dict = {
            "x": xs,
            "y": ys,
            "longitude": longitudes,
            "latitude": latitudes,
            "date": np.array([date.isoformat()] * xs.size),
        }
        for name, arr in zip(valid_band_names, band_arrays):
            data_dict[name] = arr

        table = pl.DataFrame(data_dict)

    return table
