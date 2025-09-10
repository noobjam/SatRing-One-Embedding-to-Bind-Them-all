import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.plot import reshape_as_image
import numpy as np
from torchvision import transforms

class SatelliteDataset(Dataset):
    """
    Custom Dataset for loading satellite imagery from Sentinel-1/2 (including .SAFE format), Landsat 8/9, TIFF, and compatible data.
    Loads from preprocessed products in data/processed (no normalization/scaling here).
    Can handle both flat TIFF files and raw .SAFE format folders (containing .jp2 bands).
    """
    def __init__(self, root_dir="data/processed", transform=None, jp2_bands=("B04", "B03", "B02")):
        """
        Args:
            root_dir (string): Directory with all the TIFF files and/or .SAFE folders (default: data/processed).
            transform (callable, optional): Optional transform to be applied on a sample.
            jp2_bands (tuple of str): For .SAFE Sentinel-2, which bands to load (default: RGB).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.jp2_bands = jp2_bands
        tiff_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.tif', '.tiff'))]
        safe_dirs = [f for f in os.listdir(root_dir) if f.endswith('.SAFE') and os.path.isdir(os.path.join(root_dir, f))]
        # Enumerate both TIFFs and .SAFEs
        self.entries = []
        for f in tiff_files:
            self.entries.append({'type':'tiff', 'path':f})
        for d in safe_dirs:
            self.entries.append({'type':'safe', 'path':d})

    def __len__(self):
        return len(self.entries)

    def _load_safe_bands(self, safe_dir):
        """
        Loads selected bands from a .SAFE directory (handles .jp2 or .tif bands, and collects QI bands).
        Returns
            image: np.ndarray [H, W, len(bands)] with float32 data
            band_paths: actual disk paths to band files used (for each band, .jp2 or .tif)
            qi_band_paths: dict of QI band type to file path (found in IMG_DATA)
        """
        safe_path = os.path.join(self.root_dir, safe_dir)
        img_data_dirs = glob.glob(os.path.join(safe_path, "GRANULE", "*", "IMG_DATA*"))
        if not img_data_dirs:
            raise RuntimeError(f"Could not find IMG_DATA folder in {safe_path}")
        img_data_dir = img_data_dirs[0]
        images = []
        band_files = []
        # Search for user-requested bands as .jp2 or .tif
        for b in self.jp2_bands:
            jp2_pattern = os.path.join(img_data_dir, f"*_{b}.jp2")
            tif_pattern = os.path.join(img_data_dir, f"*_{b}.tif")
            found = glob.glob(jp2_pattern) + glob.glob(tif_pattern)
            if not found:
                raise FileNotFoundError(f"No .jp2 or .tif band file for {b} in {img_data_dir}")
            in_file = found[0]
            with rasterio.open(in_file) as src:
                img = src.read(1).astype(np.float32)
                images.append(img)
                band_files.append(in_file)
        # QI bands (all in IMG_DATA starting with QI_*, jpg2 or tif)
        qi_band_paths = {}
        qi_files_jp2 = glob.glob(os.path.join(img_data_dir, "QI_*.jp2"))
        qi_files_tif = glob.glob(os.path.join(img_data_dir, "QI_*.tif"))
        for qf in qi_files_jp2 + qi_files_tif:
            qi_name = os.path.splitext(os.path.basename(qf))[0]
            qi_band_paths[qi_name] = qf
        # Stack bands as H x W x C
        stacked = np.stack(images, axis=-1)
        return stacked, band_files, qi_band_paths

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        entry = self.entries[idx]

        if entry['type'] == 'tiff':
            file_path = os.path.join(self.root_dir, entry['path'])
            with rasterio.open(file_path) as src:
                image = reshape_as_image(src.read()).astype(np.float32)
            sample = {'image': image, 'filename': entry['path']}
        elif entry['type'] == 'safe':
            image, band_files, qi_band_paths = self._load_safe_bands(entry['path'])
            sample = {
                'image': image,
                'filename': entry['path'],
                'bands': band_files,
                'qi_band_paths': qi_band_paths,
            }
        else:
            raise ValueError(f"Unknown entry type: {entry['type']}")

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

def get_dataloader(root_dir="data/processed", batch_size=32, shuffle=True, num_workers=4, transform=None):
    """
    Creates a DataLoader for the satellite dataset.
    
    Args:
        root_dir (string): Directory with processed files (default: data/processed).
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of workers for data loading.
        transform (callable, optional): Transforms to apply.
    
    Returns:
        DataLoader: PyTorch DataLoader for batches.
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Add any additional transforms, e.g., for SSL augmentations
        ])
    
    dataset = SatelliteDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# Example usage (commented out)
# if __name__ == "__main__":
#     root_dir = "data/raw/"
#     dataloader = get_dataloader(root_dir, batch_size=4)
#     for batch in dataloader:
#         print(batch['image'].shape)
#         break
