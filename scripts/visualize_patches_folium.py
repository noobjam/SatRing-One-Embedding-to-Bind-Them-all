"""
Generate interactive Folium map for Rwanda patches.

Usage:
    python scripts/visualize_patches_folium.py
"""

import geopandas as gpd
import folium
from folium import plugins

# Load patches
patches = gpd.read_file('patch_metadata/patches_rwanda_stratified.geojson')

# Convert to WGS84 for Folium
patches = patches.to_crs(epsg=4326)

# Create map centered on Rwanda
rwanda_center = [-1.95, 29.87]
m = folium.Map(
    location=rwanda_center,
    zoom_start=9,
    tiles='OpenStreetMap'
)

# Color mapping for classes
class_colors = {
    'Forest': '#228B22',
    'Cropland': '#FFD700',
    'Grassland': '#90EE90',
    'Urban': '#FF6347',
    'Settlement': '#FF6347',
    'Water': '#4169E1',
    'Wetland': '#00CED1',
    'Bare': '#D2691E',
}

complexity_colors = {
    'homogeneous': '#3498db',
    'moderate': '#f39c12',
    'complex': '#e74c3c'
}

# Add patches to map
for idx, row in patches.iterrows():
    # Get color based on complexity (or change to dominant_class)
    color = complexity_colors.get(row['complexity'], '#95a5a6')
    
    # Create popup with metadata
    popup_html = f"""
    <b>Patch {idx}</b><br>
    Tile: {row['tile']}<br>
    Class: {row['dominant_class']}<br>
    Complexity: {row['complexity']}<br>
    Entropy: {row['entropy']:.2f}<br>
    OSM: {row.get('osm_tag', 'N/A')}
    """
    
    # Add rectangle
    folium.Rectangle(
        bounds=[
            [row.geometry.bounds[1], row.geometry.bounds[0]],  # SW corner
            [row.geometry.bounds[3], row.geometry.bounds[2]]   # NE corner
        ],
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.4,
        weight=1,
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Add legend
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 200px; height: 120px; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 10px">
<p><b>Complexity</b></p>
<p><span style="color:#3498db">■</span> Homogeneous</p>
<p><span style="color:#f39c12">■</span> Moderate</p>
<p><span style="color:#e74c3c">■</span> Complex</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Add fullscreen option
plugins.Fullscreen().add_to(m)

# Save
output_path = 'patch_metadata/patches_rwanda_map.html'
m.save(output_path)
print(f"✓ Saved interactive map to {output_path}")
print(f"  Open in browser to explore {len(patches)} patches")
