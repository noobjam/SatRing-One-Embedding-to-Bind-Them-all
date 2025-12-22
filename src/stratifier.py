import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import numpy as np



TARGET_RES = 10.0



class Stratifier:
    def __init__(self):
        pass