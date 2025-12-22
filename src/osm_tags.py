"""
OSM land use tag mapping for Rwanda.

Based on OpenStreetMap landuse/natural tags commonly found in Rwanda.
"""

# OSM land use tags to integer encoding
OSM_TAG_MAPPING = {
    'unknown': 0,
    'forest': 1,
    'farmland': 2,
    'residential': 3,
    'commercial': 4,
    'industrial': 5,
    'grass': 6,
    'scrub': 7,
    'wetland': 8,
    'water': 9,
    'bare_ground': 10,
    'urban': 11,
    'agriculture': 12,  # Alias for farmland
}

# Reverse mapping
OSM_INT_TO_TAG = {v: k for k, v in OSM_TAG_MAPPING.items()}

def encode_osm_tag(tag_str):
    """
    Convert OSM tag string to integer.
    
    Args:
        tag_str: OSM landuse/natural tag (e.g., 'forest', 'farmland')
    
    Returns:
        int: Encoded tag (0 for unknown)
    """
    if tag_str is None or tag_str == '':
        return 0
    
    # Normalize
    tag_str = str(tag_str).lower().strip()
    
    # Direct match
    if tag_str in OSM_TAG_MAPPING:
        return OSM_TAG_MAPPING[tag_str]
    
    # Aliases
    if 'farm' in tag_str or 'crop' in tag_str:
        return OSM_TAG_MAPPING['farmland']
    if 'tree' in tag_str or 'wood' in tag_str:
        return OSM_TAG_MAPPING['forest']
    if 'reside' in tag_str or 'housing' in tag_str:
        return OSM_TAG_MAPPING['residential']
    if 'water' in tag_str or 'river' in tag_str or 'lake' in tag_str:
        return OSM_TAG_MAPPING['water']
    
    # Unknown
    return 0

def decode_osm_tag(tag_int):
    """Convert integer tag back to string."""
    return OSM_INT_TO_TAG.get(tag_int, 'unknown')
