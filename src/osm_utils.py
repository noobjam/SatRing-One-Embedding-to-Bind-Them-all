"""
Utilities for integrating OpenStreetMap data with satellite imagery.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio import features
from pathlib import Path
import osmnx as ox

class OSMRasterizer:
    """
    Convert OSM vector data to raster masks aligned with satellite imagery.
    """
    
    def __init__(self, cache_dir="osm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_osm_features(self, bbox, tags):
        """
        Download OSM features for a bounding box.
        
        Args:
            bbox: (minx, miny, maxx, maxy) in WGS84
            tags: dict of OSM tags, e.g., {'building': True, 'highway': True}
        
        Returns:
            GeoDataFrame with OSM features
        """
        try:
            # Download from OSM
            gdf = ox.features_from_bbox(
                bbox=bbox,  # (north, south, east, west)
                tags=tags
            )
            return gdf
        except Exception as e:
            print(f"Error downloading OSM data: {e}")
            return None
    
    def rasterize_features(self, gdf, transform, shape, feature_type='all'):
        """
        Convert vector features to raster mask.
        
        Args:
            gdf: GeoDataFrame with features
            transform: Affine transform (from rasterio)
            shape: (height, width) of output raster
            feature_type: 'roads', 'buildings', 'water', or 'all'
        
        Returns:
            numpy array [H, W] with binary mask
        """
        if gdf is None or len(gdf) == 0:
            return np.zeros(shape, dtype=np.uint8)
        
        # Filter by feature type
        if feature_type == 'roads':
            gdf = gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])]
        elif feature_type == 'buildings':
            gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            gdf = gdf[gdf.get('building', False) == True]
        elif feature_type == 'water':
            gdf = gdf[gdf.get('natural', '') == 'water']
        
        # Rasterize
        shapes = [(geom, 1) for geom in gdf.geometry]
        
        if len(shapes) == 0:
            return np.zeros(shape, dtype=np.uint8)
        
        raster = features.rasterize(
            shapes=shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        
        return raster
    
    def create_osm_stack(self, bbox, transform, shape):
        """
        Create multi-channel OSM raster stack.
        
        Args:
            bbox: (minx, miny, maxx, maxy)
            transform: Affine transform
            shape: (height, width)
        
        Returns:
            numpy array [C, H, W] with C channels for different OSM features
        """
        # Download different feature types
        tags_roads = {'highway': True}
        tags_buildings = {'building': True}
        tags_water = {'natural': 'water'}
        tags_landuse = {'landuse': True}
        
        # Get features
        roads = self.get_osm_features(bbox, tags_roads)
        buildings = self.get_osm_features(bbox, tags_buildings)
        water = self.get_osm_features(bbox, tags_water)
        landuse = self.get_osm_features(bbox, tags_landuse)
        
        # Rasterize each
        roads_raster = self.rasterize_features(roads, transform, shape, 'roads')
        buildings_raster = self.rasterize_features(buildings, transform, shape, 'buildings')
        water_raster = self.rasterize_features(water, transform, shape, 'water')
        
        # Stack
        osm_stack = np.stack([roads_raster, buildings_raster, water_raster], axis=0)
        
        return osm_stack.astype(np.float32)


def get_osm_tags_for_location(lat, lon, radius=1000):
    """
    Get OSM tags for a location (for weak supervision).
    
    Args:
        lat, lon: Location coordinates
        radius: Search radius in meters
    
    Returns:
        dict of OSM tags
    """
    try:
        # Get features around point
        tags = {'landuse': True, 'natural': True, 'building': True}
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius)
        
        if len(gdf) == 0:
            return {}
        
        # Extract most common tags
        landuse_tags = gdf['landuse'].dropna().value_counts()
        natural_tags = gdf['natural'].dropna().value_counts()
        
        result = {}
        if len(landuse_tags) > 0:
            result['landuse'] = landuse_tags.index[0]
        if len(natural_tags) > 0:
            result['natural'] = natural_tags.index[0]
        
        return result
    
    except Exception as e:
        print(f"Error getting OSM tags: {e}")
        return {}


# Example usage
if __name__ == "__main__":
    # Example: Rwanda bounding box
    bbox = (28.8, -2.9, 30.9, -1.0)  # (minx, miny, maxx, maxy)
    
    # Create rasterizer
    rasterizer = OSMRasterizer()
    
    # For a specific tile (you'd get this from your satellite data)
    from rasterio.transform import from_bounds
    transform = from_bounds(*bbox, 224, 224)
    
    # Create OSM stack
    osm_stack = rasterizer.create_osm_stack(bbox, transform, (224, 224))
    
    print(f"OSM stack shape: {osm_stack.shape}")
    print(f"Roads pixels: {osm_stack[0].sum()}")
    print(f"Buildings pixels: {osm_stack[1].sum()}")
    print(f"Water pixels: {osm_stack[2].sum()}")
