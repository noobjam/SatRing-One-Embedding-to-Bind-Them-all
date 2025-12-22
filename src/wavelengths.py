import torch

# Wavelengths in micrometers
# Assuming 7 common bands: Coastal, Blue, Green, Red, NIR, SWIR1, SWIR2

S2_WAVELENGTHS = torch.tensor([
    0.443, # B1 Coastal
    0.490, # B2 Blue
    0.560, # B3 Green
    0.665, # B4 Red
    0.865, # B8 NIR (Broad) or B8A? Usually B8A is better for comparison. Let's use 0.865.
    1.610, # B11 SWIR1
    2.190, # B12 SWIR2
], dtype=torch.float32)

LS_WAVELENGTHS = torch.tensor([
    0.443, # B1 Coastal
    0.482, # B2 Blue
    0.561, # B3 Green
    0.655, # B4 Red
    0.865, # B5 NIR
    1.610, # B6 SWIR1
    2.200, # B7 SWIR2
], dtype=torch.float32)

SENSOR_WAVELENGTHS = {
    0: S2_WAVELENGTHS, # S2
    1: LS_WAVELENGTHS, # LS
}
