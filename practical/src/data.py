"""
PyTorch Dataset for VIRVS Virtual Staining (Brightfield -> Fluorescent Infection Reporter)
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class VIRVSDataset(Dataset):
    """
    Dataset class loading paired Brightfield (input y) and Fluorescence (target x) micrographs.
    """
    def __init__(self, root_dir, split="train", normalize_range=(-1.0, 1.0), transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.normalize_range = normalize_range
        self.transform = transform
        
        self.bf_dir = os.path.join(root_dir, split, "brightfield")
        self.fluo_dir = os.path.join(root_dir, split, "fluorescence")
        
        self.bf_files = sorted(glob.glob(os.path.join(self.bf_dir, "*.png")))
        self.fluo_files = sorted(glob.glob(os.path.join(self.fluo_dir, "*.png")))
        
        assert len(self.bf_files) > 0, f"No brightfield images found in {self.bf_dir}"
        assert len(self.bf_files) == len(self.fluo_files), (
            f"Mismatch between BF ({len(self.bf_files)}) and Fluo ({len(self.fluo_files)}) images"
        )
        
    def __len__(self):
        return len(self.bf_files)
        
    def __getitem__(self, idx):
        bf_path = self.bf_files[idx]
        fluo_path = self.fluo_files[idx]
        
        # Load as float32 in [0, 1]
        bf_img = np.array(Image.open(bf_path).convert("L"), dtype=np.float32) / 255.0
        fluo_img = np.array(Image.open(fluo_path).convert("L"), dtype=np.float32) / 255.0
        
        # Add channel dimension: [1, H, W]
        bf_tensor = torch.from_numpy(bf_img).unsqueeze(0)
        fluo_tensor = torch.from_numpy(fluo_img).unsqueeze(0)
        
        # Scale range if requested (e.g. [-1, 1] for Pix2Pix GAN, [0, 1] for U-Net)
        if self.normalize_range == (-1.0, 1.0):
            bf_tensor = bf_tensor * 2.0 - 1.0
            fluo_tensor = fluo_tensor * 2.0 - 1.0
            
        return {
            "brightfield": bf_tensor,
            "fluorescence": fluo_tensor,
            "bf_filename": os.path.basename(bf_path)
        }
