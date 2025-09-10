from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType
from rasterio.errors import RasterioIOError
from rasterio.warp import Resampling, reproject
from tqdm import tqdm
import tifffile
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import shutil
# ---------------------------------------------------------------------------
# Spark UDFs and helpers
# ---------------------------------------------------------------------------

result_schema = StructType(
    [
        StructField("x", ArrayType(IntegerType()), nullable=False),
        StructField("y", ArrayType(IntegerType()), nullable=False),
    ]
)


@udf(result_schema)
def convert_extent_to_coords_udf(extent, size, step):
    """
    Convert raster extent and size to coordinate arrays.

    Parameters
    ----------
    extent : pyrasterframes Extent
        Object with bounds representing xmin, ymin, xmax, ymax.
    size : tuple[int, int]
        (width, height) in pixels.
    step : int
        Step size to offset bounds by half the pixel size.

    Returns
    -------
    tuple[list[int], list[int]]
        Flattened meshgrid coordinates (x, y).
    """
    bounds = list(extent.bounds)
    bounds = [
        bounds[0] + step // 2,
        bounds[1] + step // 2,
        bounds[2] - step // 2,
        bounds[3] - step // 2,
    ]
    xmin, ymin, xmax, ymax = bounds
    cols, rows = np.meshgrid(
        np.linspace(xmin, xmax, size[0], dtype=int),
        np.linspace(ymax, ymin, size[1], dtype=int),
    )
    xs = np.array(cols).flatten().tolist()
    ys = np.array(rows).flatten().tolist()
    return xs, ys


def get_metadata(tiff_path: str) -> tuple[int, List[int]]:
    """
    Read basic TIFF metadata using tifffile.

    Returns
    -------
    (pixel_scale_x, band_indices)
      pixel_scale_x: int
      band_indices: list of sample indices [0..SamplesPerPixel-1]
    """
    with tifffile.TiffFile(tiff_path) as tif:
        metadata = tif.pages[0].tags
        return int(metadata["ModelPixelScaleTag"].value[0]), list(
            range(metadata["SamplesPerPixel"].value)
        )


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _resample_file_worker(
    src_path: str,
    ref_meta: dict,
    factor: float,
    continuous_resampling: Resampling,
    scl_identifiers: Sequence[str],
    out_path: str,
) -> None:
    """
    Multiprocessing-safe worker to resample a single raster to a reference grid.
    """
    src_p = Path(src_path)
    out_p = Path(out_path)

    is_scl = any(token.upper() in src_p.name.upper() for token in scl_identifiers)

    with rasterio.open(src_p) as src:
        arr = src.read(1)

        if not is_scl:
            arr = arr.astype("float32", copy=False)
            if np.isfinite(arr).any() and np.nanmax(arr) > 1.5:
                arr = arr / float(factor)
        else:
            arr = arr.astype("int16", copy=False)

        needs_reproject = any(
            [
                src.transform != ref_meta["transform"],
                src.crs != ref_meta["crs"],
                src.width != ref_meta["width"],
                src.height != ref_meta["height"],
            ]
        )

        if needs_reproject:
            dst = np.empty((ref_meta["height"], ref_meta["width"]), dtype=arr.dtype)
            reproject(
                source=arr,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_meta["transform"],
                dst_crs=ref_meta["crs"],
                resampling=(Resampling.nearest if is_scl else continuous_resampling),
            )
            arr = dst

        out_meta = ref_meta.copy()
        out_meta.update(
            {
                "dtype": arr.dtype,
                "count": 1,
                "driver": "GTiff",
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
            }
        )

        out_p.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_p, "w", **out_meta) as dst:
            dst.write(arr, 1)
            try:
                tags = src.tags()
                if tags:
                    dst.update_tags(**tags)
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Mosaic/resampling class
# ---------------------------------------------------------------------------


class RFMosaic:
    """
    SAFE/GeoTIFF resampling utility focused on consistent alignment to a reference tile.

    Responsibilities
    - Discover product directories under a root path.
    - Identify a reference tile (e.g., B02 10m) and cache its metadata.
    - Resample/reproject source rasters to match the reference grid.
    - Scale reflectance for non-mask layers when needed.
    - Write outputs alongside sources using a safe suffix (no destructive overwrite).

    Notes
    - Uses nearest resampling for mask-like layers (SCL/QA), bilinear (default) for imagery.
    - Writes tiled, compressed GeoTIFFs for performance.

    Parameters
    ----------
    spark_session : SparkSession 
        Stored for potential distributed workflows (not required in this class).
    path : str
        Root path containing product folders (e.g., "S2*", "LS*").
    scl_path : Optional[str]
        Optional path for SCL; not directly used but kept for interface compatibility.
    include_patterns : Sequence[str]
        Folder name prefixes to include (e.g. ("S2", "LS")).
    reference_tile : Optional[str]
        Explicit path to a raster to act as reference. If None, auto-discovered.
    output_dir : Optional[str]
        If provided, outputs can be directed elsewhere. By default, writes beside inputs.
    target_band_pattern : str
        Filename fragment used to locate a default reference (e.g., "B02_10m").
    """

    def __init__(
        self,
        spark_session: SparkSession,
        path: str,
        scl_path: Optional[str] = None,
        include_patterns: Sequence[str] = ("S2", "LS"),
        reference_tile: Optional[str] = None,
        output_dir: Optional[str] = None,
        target_band_pattern: str = "B02_10m",
    ) -> None:
        self.spark_session = spark_session
        self.root = Path(path).expanduser().resolve()
        self.scl_path = Path(scl_path).resolve() if scl_path else None
        self.include_patterns = tuple(include_patterns)
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.target_band_pattern = target_band_pattern

        self.products: List[Path] = self._discover_products(self.root,self.include_patterns)
        if not self.products:
            logger.warning(f"No product folders found under: {self.root}")

        # Reference tile path
        self.reference_tile: Optional[Path] = (
            Path(reference_tile).resolve() if reference_tile else self._find_reference_tile()
        )
        if self.reference_tile is None:
            logger.error(
                "Could not locate a reference tile. Provide reference_tile or ensure "
                f"a file containing '{self.target_band_pattern}' exists under products."
            )
        else:
            logger.info(f"Reference tile: {self.reference_tile}")

        # Cached reference metadata (GDAL profile)
        self.reference_meta: Optional[dict] = None

    # ------------------------------- Public API -------------------------------
    
    
    
    def _stack_bands(self):
        products = self._discover_products(self.root, self.include_patterns)

        subdir_filter = "IMG_DATA"
        files = []

        for product in products:
            img_data_dirs = [d for d in product.rglob("*") if d.is_dir() and subdir_filter in d.name]

            for img_dir in img_data_dirs:
                for p in img_dir.glob("*"):  
                    if not p.is_file(): # we don't care about NATIVE for now
                        continue

                    if "resampled" not in p.name.lower():
                        continue

                    file_name = str(self.root)+ '/stacked/' + product.name.replace("SAFE", "TIF")
                    # print("Found:", p.name, "→", file_name)

                    files.append(p)
                    print(f"start stacking for {file_name}")
                    os.makedirs(os.path.dirname(file_name), exist_ok=True)

                    
                    with rasterio.open(files[0]) as src0:
                        meta = src0.meta
                        
                    meta.update(count = len(files))
   

                    with rasterio.open(file_name, 'w', **meta) as dst:
                        for id, layer in enumerate(files, start=1):
                            with rasterio.open(layer) as src1:
                                dst.write_band(id, src1.read(1))
        

        
    def mosaic(self,
              reasmple:bool = True,
              stack:bool = True):
        if resample:
            self.resample()
        if stack:
            self.stack()
        

            
        #resample
        slef.stack()
        #stack
        #actual mosaicking
        pass
    
    def resample(
        self,
        factor: float = 10000.0,
        resampling: Resampling = Resampling.bilinear,
        scl_identifier: str = "SCL",
        overwrite: bool = False,
        extensions: Sequence[str] = (".tif", ".tiff", ".jp2"),
        subdir_filter: str = "IMG_DATA",
    ) -> None:
        """
        Resample/reproject all matching rasters to the reference grid.

        Parameters
        ----------
        factor : float
            Scale factor applied to non-mask rasters if values appear to be unscaled reflectance.
        resampling : Resampling
            Resampling method for continuous imagery (non-SCL/QA).
        scl_identifiers : Sequence[str]
            Identifiers used to detect mask-like rasters (nearest resampling).
        overwrite : bool
            If True, overwrite existing outputs with the same name. Otherwise write with a suffix.
        extensions : Sequence[str]
            File extensions to consider as source rasters.
        subdir_filter : str
            Only process files whose parent path contains this token (e.g., SAFE IMG_DATA).
        """
        if self.reference_tile is None:
            raise RuntimeError("Reference tile is required but was not found.")
        
        
        self._ensure_reference_meta_loaded()
        
        self._move_scl_files(extensions=extensions)
        
        files = self._list_source_files(self.products,extensions=extensions, subdir_filter=subdir_filter)
        if not files:
            logger.warning("No source rasters found to resample.")
            return

        logger.info(f"Resampling {len(files)} file(s) to match: {self.reference_tile}")

        # Pre-derive output paths and skip existing when not overwriting
        tasks: List[tuple[Path, Path]] = []
        for src_path in files:
            out_path = self._derive_output_path(src_path, overwrite=overwrite)
            if out_path.exists() and not overwrite:
                logger.debug(f"Output exists, skipping: {out_path}")
                continue
            tasks.append((src_path, out_path))

        if not tasks:
            logger.info("Nothing to process after filtering existing outputs.")
            return

        workers = os.cpu_count() or 1
        logger.info(f"Using {workers} process(es)")

        # Fan-out to processes; each worker opens files independently (GDAL-safe)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _resample_file_worker,
                    str(src_path),
                    self.reference_meta,  # picklable (Affine, CRS)
                    float(factor),
                    resampling,
                    tuple(scl_identifier),
                    str(out_path),
                )
                for src_path, out_path in tasks
            ]

            with tqdm(total=len(futures), desc="Resampling files", leave=False) as pbar:
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        # Log and continue with other files
                        logger.exception(f"Failed to resample file: {e}")
                    finally:
                        pbar.update(1)

    # ------------------------------ Internal API ------------------------------
    
    def _move_scl_files(self, extensions: Sequence[str]) -> None:
        """Move SCL files from scl_path to appropriate product directories."""
        products = self._discover_products(self.scl_path,  patterns = ("S2", "LC"))
        
        for product in products:            
            product_name = product.name.upper()
            if product_name.startswith('S2'):
                scl_identifier = "SCL_20"
                scl_file = self._list_source_files([product], extensions, subdir_filter=scl_identifier)
                if len(scl_file) > 0:
                    scl_file_splitted = str(scl_file[0]).split('/')[8].replace('MSIL1C', 'MSIL2F').split('_')
                    folder_prefix = '_'.join([scl_file_splitted[0], scl_file_splitted[1], scl_file_splitted[2]])
                else: 
                    continue
            elif product_name.startswith('LC'):
                scl_identifier = "SCL"
                scl_file = self._list_source_files([product], extensions, subdir_filter=scl_identifier)
                if len(scl_file) > 0:
                    scl_file_splitted = str(scl_file[0]).split('/')[8].replace('LC0', 'LS').replace('L1TP', 'OLIL2F').split('_')
                    folder_prefix = '_'.join([scl_file_splitted[0], scl_file_splitted[1], scl_file_splitted[3]])
                else: 
                    continue
            else:
                logger.warning(f"Unknown product type for {product.name}, skipping")
                continue
            # Find the unique folder matching prefix + placeholder
            dest_root = list(list(self.root.glob(f"{folder_prefix}*"))[0].rglob("IMG_DATA"))[0]
            shutil.copy(scl_file[0], dest_root)

    def _discover_products(self,root:str, patterns: Sequence[str]) -> List[Path]:
        """Return product directories under root matching any of the provided prefixes."""
        if not root.exists():
            logger.error(f"Root path does not exist: {root}")
            return []
        products: List[Path] = []
        for child in root.iterdir():
            if child.is_dir() and any(child.name.startswith(p) for p in patterns):
                products.append(child)
        products.sort()
        return products

    def _find_reference_tile(self) -> Optional[Path]:
        """
        Try to find a reference tile by searching for a filename containing target_band_pattern.
        Prefer a GeoTIFF, fallback to JP2.
        """
        candidates: List[Path] = []
        for product in self.products:
            for p in product.rglob("*"):
                name_low = p.name.lower()
                if self.target_band_pattern.lower() in name_low and (
                    name_low.endswith(".tif") or name_low.endswith(".jp2")
                ):
                    candidates.append(p)
        if not candidates:
            return None
        # Prefer TIF over JP2 if both exist
        candidates.sort(key=lambda p: 0 if p.suffix.lower() in (".tif", ".tiff") else 1)
        return candidates[0]

    def _ensure_reference_meta_loaded(self) -> None:
        """Load reference raster metadata if not already loaded."""
        if self.reference_meta is not None:
            return
        if self.reference_tile is None:
            raise RuntimeError("Reference tile not set.")
        try:
            with rasterio.open(self.reference_tile) as ref:
                self.reference_meta = ref.meta.copy()
        except RasterioIOError as e:
            raise RuntimeError(f"Failed to open reference tile: {self.reference_tile}") from e

    def _list_source_files(
        self,
        products : List[str],
        extensions: Sequence[str],
        subdir_filter: Optional[str] = None,
    ) -> List[Path]:
        """
        Collect source raster files under product folders.

        Filters to files whose parent path contains subdir_filter when provided
        (useful for SAFE IMG_DATA).
        """
        files: List[Path] = []
        for product in products:
            for p in product.rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in [ext.lower() for ext in extensions]:
                    continue
                if subdir_filter and subdir_filter not in str(p):
                    continue
                # Skip obvious previews/TCI if desired later; for now include all matching rasters.
                files.append(p)
        files.sort()
        return files

    @staticmethod
    def _is_scl_like(path: Path, identifiers: Sequence[str]) -> bool:
        """Heuristic to detect mask-like rasters (e.g., SCL/QA)."""
        name_up = path.name.upper()
        return any(token.upper() in name_up for token in identifiers)

    def _derive_output_path(self, src_path: Path, overwrite: bool) -> Path:
        """
        Determine output path for a processed raster.

        - If output_dir is set, write there mirroring filename.
        - Otherwise, write next to source with '.resampled.tif' suffix.
        - Overwrite in-place if overwrite=True and output would be same extension.
        """
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.output_dir / f"{src_path.stem}.tif"
        else:
            out_path = src_path.parent / f"{src_path.stem}.resampled.tif"

        if overwrite:
            # If overwriting, write to the original path extension as GeoTIFF
            out_path = src_path.with_suffix(".tif")
        return out_path

    def _resample_file(
        self,
        src_path: Path,
        factor: float,
        continuous_resampling: Resampling,
        scl_identifiers: Sequence[str],
        overwrite: bool,
    ) -> None:
        """Resample/reproject a single raster to the reference grid."""
        assert self.reference_meta is not None, "Reference metadata must be loaded first."
        ref_meta = self.reference_meta
        if 'resampled' in str(src_path):
            logger.debug(f"Output exists, skipping: {src_path}")
            return
        out_path = self._derive_output_path(src_path, overwrite=overwrite)
        if out_path.exists() and not overwrite:
            logger.debug(f"Output exists, skipping: {out_path}")
            return

        is_scl = self._is_scl_like(src_path, scl_identifiers)

        with rasterio.open(src_path) as src:
            # Read first band; extend if multi-band support is needed
            arr = src.read(1)

            # Normalize reflectance-like layers to float range if likely unscaled
            if not is_scl:
                arr = arr.astype("float32", copy=False)
                if np.isfinite(arr).any() and np.nanmax(arr) > 1.5:
                    arr = arr / float(factor)
            else:
                # Keep mask-like rasters as integer class labels
                # Use int16 to preserve classes while minimizing size
                arr = arr.astype("int16", copy=False)

            needs_reproject = any(
                [
                    src.transform != ref_meta["transform"],
                    src.crs != ref_meta["crs"],
                    src.width != ref_meta["width"],
                    src.height != ref_meta["height"],
                ]
            )

            if needs_reproject:
                dst = np.empty((ref_meta["height"], ref_meta["width"]), dtype=arr.dtype)
                reproject(
                    source=arr,
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_meta["transform"],
                    dst_crs=ref_meta["crs"],
                    resampling=(Resampling.nearest if is_scl else continuous_resampling),
                )
                arr = dst

            # Build output profile
            out_meta = ref_meta.copy()
            out_meta.update(
                {
                    "dtype": arr.dtype,
                    "count": 1,
                    "driver": "GTiff",
                    "compress": "lzw",
                    "tiled": True,
                    "blockxsize": 512,
                    "blockysize": 512,
                }
            )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(arr, 1)
                # Best-effort: preserve non-spatial tags
                try:
                    tags = src.tags()
                    if tags:
                        dst.update_tags(**tags)
                except Exception:
                    pass

        logger.debug(f"Wrote: {out_path}")

    # --------------------------- Optional conversions --------------------------
    @staticmethod
    def convert_jp2_to_tif(
        input_path: str,
        output_path: Optional[str] = None,
        compression: str = "lzw",
        preserve_nodata: bool = True,
        **kwargs,
    ) -> Optional[str]:
        """
        Convert a JP2 file to tiled, compressed GeoTIFF using rasterio, preserving tags.

        Parameters
        ----------
        input_path : str
            Source JP2 path.
        output_path : Optional[str]
            Destination TIF path (defaults to input with .tif extension).
        compression : str
            Compression type ('lzw', 'deflate', 'jpeg', 'none').
        preserve_nodata : bool
            Propagate nodata from source if defined.
            

        Returns
        -------
        Optional[str]
            Path to output TIF on success, else None.
        """
        try:
            in_p = Path(input_path)
            out_p = Path(output_path) if output_path else in_p.with_suffix(".tif")

            with rasterio.open(in_p) as src:
                profile = src.profile.copy()
                data = src.read()

                profile.update(
                    {
                        "driver": "GTiff",
                        "compress": compression,
                        "tiled": True,
                        "blockxsize": 512,
                        "blockysize": 512,
                    }
                )
                if preserve_nodata and src.nodata is not None:
                    profile["nodata"] = src.nodata

                profile.update(kwargs)

                out_p.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(out_p, "w", **profile) as dst:
                    dst.write(data)
                    # Copy dataset and per-band tags when available
                    try:
                        dst.update_tags(**src.tags())
                        for i in range(1, src.count + 1):
                            band_tags = src.tags(i)
                            if band_tags:
                                dst.update_tags(i, **band_tags)
                    except Exception:
                        pass

            logger.info(f"Converted JP2 to TIF: {in_p} -> {out_p}")
            return str(out_p)
        except Exception as e:
            logger.exception(f"Error converting {input_path}: {e}")
            return None
