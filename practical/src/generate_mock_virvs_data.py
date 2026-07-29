"""
Synthetic Mock Data Generator for VIRVS (Virus Infection Reporter Virtual Staining)
Creates paired Brightfield (input y) and Fluorescent Infection Reporter (target x) 
micrographs representing different virus infection stages (uninfected, early, late).
"""

import os
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image


def generate_single_cell(shape=(128, 128), center=(64, 64), radius=20, eccentricity=0.8):
    """Generate cell membrane mask and intracellular texture."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt(((x - center[1]) / eccentricity)**2 + (y - center[0])**2)
    cell_mask = dist_from_center <= radius
    return cell_mask.astype(np.float32)


def generate_virvs_pair(image_size=(256, 256), num_cells=15, infection_rate=0.6, seed=None):
    """
    Generate a paired brightfield micrograph and infection reporter fluorescence image.
    
    Parameters:
    -----------
    image_size : tuple
        (height, width) of output images.
    num_cells : int
        Number of cells to render.
    infection_rate : float
        Fraction of infected cells expressing reporter fluorescence (0.0 to 1.0).
    seed : int, optional
        Random seed for reproducibility.
        
    Returns:
    --------
    brightfield : np.ndarray (H, W) float32 in [0, 1]
    fluorescence : np.ndarray (H, W) float32 in [0, 1]
    cell_masks : np.ndarray (num_cells, H, W) bool
    """
    if seed is not None:
        np.random.seed(seed)
        
    H, W = image_size
    brightfield = np.ones((H, W), dtype=np.float32) * 0.75  # Bright background
    fluorescence = np.zeros((H, W), dtype=np.float32)
    cell_masks = []
    
    # Grid cell placement to avoid massive overlaps
    grid_side = int(np.ceil(np.sqrt(num_cells)))
    x_coords = np.linspace(35, W - 35, grid_side)
    y_coords = np.linspace(35, H - 35, grid_side)
    centers = []
    for yc in y_coords:
        for xc in x_coords:
            centers.append((yc + np.random.uniform(-10, 10), xc + np.random.uniform(-10, 10)))
    np.random.shuffle(centers)
    centers = centers[:num_cells]
    
    for i, center in enumerate(centers):
        radius = np.random.uniform(16, 25)
        eccentricity = np.random.uniform(0.7, 1.3)
        cell_mask = generate_single_cell((H, W), center, radius, eccentricity)
        
        # Smooth cell mask boundary
        smooth_cell = ndimage.gaussian_filter(cell_mask, sigma=1.5)
        cell_masks.append(cell_mask > 0.5)
        
        # 1. Render Brightfield channel (refractive index changes, phase contrast boundaries)
        cell_interior = ndimage.gaussian_filter(cell_mask, sigma=3.0)
        boundary = ndimage.laplace(smooth_cell)
        
        brightfield -= cell_interior * 0.25
        brightfield += boundary * 0.15
        
        # 2. Render Virus Infection Reporter Fluorescence channel
        is_infected = np.random.rand() < infection_rate
        if is_infected:
            # Viral replication factories / GFP intensity centered in cytoplasm/nucleus
            infection_intensity = np.random.uniform(0.4, 1.0)
            noise_texture = np.random.rand(H, W) * 0.2
            cell_fluo = smooth_cell * infection_intensity * (0.8 + noise_texture)
            cell_fluo = ndimage.gaussian_filter(cell_fluo, sigma=1.0)
            fluorescence += cell_fluo

    # Add realistic illumination gradient & camera noise
    y_grid, x_grid = np.ogrid[:H, :W]
    vignette = 1.0 - 0.15 * (((x_grid - W/2)/(W/2))**2 + ((y_grid - H/2)/(H/2))**2)
    brightfield *= vignette
    
    # Add shot noise
    brightfield += np.random.normal(0, 0.02, (H, W))
    fluorescence += np.random.normal(0, 0.01, (H, W))
    
    # Clip to valid range [0, 1]
    brightfield = np.clip(brightfield, 0.0, 1.0).astype(np.float32)
    fluorescence = np.clip(fluorescence, 0.0, 1.0).astype(np.float32)
    cell_masks = np.array(cell_masks, dtype=bool)
    
    return brightfield, fluorescence, cell_masks


def create_dataset_directory(output_dir, num_train=40, num_val=10, image_size=(256, 256)):
    """Generate synthetic train/val directory structure matching VIRVS layout."""
    os.makedirs(os.path.join(output_dir, "train", "brightfield"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "fluorescence"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "brightfield"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "fluorescence"), exist_ok=True)
    
    print(f"[+] Generating {num_train} train pairs & {num_val} val pairs in {output_dir}...")
    
    for split, count in [("train", num_train), ("val", num_val)]:
        for idx in range(count):
            seed = (1000 if split == "train" else 5000) + idx
            bf, fluo, _ = generate_virvs_pair(image_size=image_size, seed=seed)
            
            # Save as 16-bit PNG or 8-bit PNG images
            bf_img = Image.fromarray((bf * 255).astype(np.uint8))
            fluo_img = Image.fromarray((fluo * 255).astype(np.uint8))
            
            bf_img.save(os.path.join(output_dir, split, "brightfield", f"frame_{idx:04d}.png"))
            fluo_img.save(os.path.join(output_dir, split, "fluorescence", f"frame_{idx:04d}.png"))
            
    print(f"[+] Data generation complete!")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data", "mock_virvs"))
    create_dataset_directory(data_dir, num_train=30, num_val=10)
