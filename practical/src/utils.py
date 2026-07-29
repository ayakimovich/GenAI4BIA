"""
Visualization Utilities for Virtual Staining Benchmarking
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_virtual_staining_comparison(
    brightfield, target_fluo, pred_unet=None, pred_pix2pix=None, 
    save_path=None, title="Virtual Staining Result"
):
    """
    Side-by-side visual comparison of input brightfield, ground truth fluorescence, 
    and model predictions (U-Net / Pix2Pix).
    """
    if isinstance(brightfield, torch.Tensor):
        brightfield = brightfield.detach().cpu().numpy()
    if isinstance(target_fluo, torch.Tensor):
        target_fluo = target_fluo.detach().cpu().numpy()
    if pred_unet is not None and isinstance(pred_unet, torch.Tensor):
        pred_unet = pred_unet.detach().cpu().numpy()
    if pred_pix2pix is not None and isinstance(pred_pix2pix, torch.Tensor):
        pred_pix2pix = pred_pix2pix.detach().cpu().numpy()

    bf_2d = np.squeeze(brightfield)
    gt_2d = np.squeeze(target_fluo)

    num_cols = 2
    if pred_unet is not None:
        num_cols += 1
    if pred_pix2pix is not None:
        num_cols += 1

    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    
    # 1. Input Brightfield
    axes[0].imshow(bf_2d, cmap="gray")
    axes[0].set_title("Input Brightfield (y)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Target Fluorescence
    axes[1].imshow(gt_2d, cmap="magma")
    axes[1].set_title("Target Reporter Fluo (x)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    curr_col = 2
    if pred_unet is not None:
        unet_2d = np.squeeze(pred_unet)
        axes[curr_col].imshow(unet_2d, cmap="magma")
        axes[curr_col].set_title("U-Net Prediction", fontsize=11, fontweight="bold")
        axes[curr_col].axis("off")
        curr_col += 1

    if pred_pix2pix is not None:
        pix_2d = np.squeeze(pred_pix2pix)
        axes[curr_col].imshow(pix_2d, cmap="magma")
        axes[curr_col].set_title("Pix2Pix Prediction", fontsize=11, fontweight="bold")
        axes[curr_col].axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        
    plt.show()
    plt.close()
