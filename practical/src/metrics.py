"""
Quantitative Evaluation Metrics for VIRVS Virtual Staining Benchmark:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- PCC (Pearson Correlation Coefficient)
- MAE (Mean Absolute Error)
- Cell-level Infection Reporter Quantification
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as skimage_ssim


def compute_mae(target, pred):
    """Mean Absolute Error (tensor or numpy)."""
    if isinstance(target, torch.Tensor):
        return torch.mean(torch.abs(target - pred)).item()
    return float(np.mean(np.abs(target - pred)))


def compute_psnr(target, pred, max_val=1.0):
    """Peak Signal-to-Noise Ratio (dB)."""
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        
    mse = np.mean((target - pred) ** 2)
    if mse == 0:
        return float('inf')
    return float(10.0 * np.log10((max_val ** 2) / mse))


def compute_ssim(target, pred, max_val=1.0):
    """Structural Similarity Index (SSIM)."""
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        
    # Ensure 2D spatial shape [H, W]
    target_2d = np.squeeze(target)
    pred_2d = np.squeeze(pred)
    
    return float(skimage_ssim(target_2d, pred_2d, data_range=max_val))


def compute_pcc(target, pred):
    """Pearson Correlation Coefficient (PCC)."""
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        
    t_flat = target.flatten()
    p_flat = pred.flatten()
    
    if np.std(t_flat) == 0 or np.std(p_flat) == 0:
        return 0.0
        
    pcc_matrix = np.corrcoef(t_flat, p_flat)
    return float(pcc_matrix[0, 1])


def compute_cell_reporter_stats(target, pred, cell_masks):
    """
    Quantify cell-level mean fluorescence intensity for viral infection analysis.
    
    Parameters:
    -----------
    target : np.ndarray or torch.Tensor [H, W]
        Ground truth fluorescence micrograph.
    pred : np.ndarray or torch.Tensor [H, W]
        Predicted virtual stain fluorescence micrograph.
    cell_masks : np.ndarray [Num_Cells, H, W] bool
        Array of single-cell binary segmentation masks.
        
    Returns:
    --------
    results : dict containing cell-level true and predicted mean reporter intensities
    """
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        
    target_2d = np.squeeze(target)
    pred_2d = np.squeeze(pred)
    
    true_intensities = []
    pred_intensities = []
    
    for k in range(cell_masks.shape[0]):
        mask = cell_masks[k]
        if np.sum(mask) > 0:
            true_val = np.mean(target_2d[mask])
            pred_val = np.mean(pred_2d[mask])
            true_intensities.append(true_val)
            pred_intensities.append(pred_val)
            
    true_intensities = np.array(true_intensities)
    pred_intensities = np.array(pred_intensities)
    
    cell_pcc = float(np.corrcoef(true_intensities, pred_intensities)[0, 1]) if len(true_intensities) > 1 else 1.0
    cell_mae = float(np.mean(np.abs(true_intensities - pred_intensities)))
    
    return {
        "true_intensities": true_intensities,
        "pred_intensities": pred_intensities,
        "cell_pcc": cell_pcc,
        "cell_mae": cell_mae
    }
