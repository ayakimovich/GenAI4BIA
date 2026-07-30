"""
Python helper script to build valid, clean Jupyter Notebooks for the practical workshop.
"""

import json
import os


def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5
    }


def code_cell(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }


def md_cell(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }


# ==============================================================================
# Notebook 01: Data Download & Preparation
# ==============================================================================
nb01 = make_nb([
    md_cell("""# Notebook 01: Data Download & Preparation (VIRVS Benchmark)

Welcome to the practical workshop on **Virus Infection Reporter Virtual Staining (VIRVS)**!

### Overview
In this notebook, we explore the dataset structure introduced in the VIRVS benchmark ([Wyrzykowska et al., 2024](https://github.com/casus/virvs)). The benchmark maps non-destructive label-free **Brightfield** micrographs ($y$) to continuous **Fluorescence** infection reporter signals ($x$).

**Key Learning Objectives**:
1. Set up the environment (with 1-click Google Colab support).
2. Understand dataset pair structure for virtual staining.
3. Generate/load paired micrographs.
4. Implement PyTorch `Dataset` & `DataLoader` pipelines."""),

    code_cell("""# ==========================================================
# 1. Google Colab Setup & Environment Setup
# ==========================================================
import sys
import os

if 'google.colab' in sys.modules:
    print("[+] Google Colab detected! Installing dependencies and setting up paths...")
    !git clone https://github.com/ayakimovich/GenAI4BIA.git /content/GenAI4BIA
    %cd /content/GenAI4BIA/practical
    !pip install -r ../requirements.txt -q
    sys.path.append(os.path.abspath("."))
else:
    print("[+] Local execution detected.")
    sys.path.append(os.path.abspath("."))"""),

    md_cell("""## 2. Generate Synthetic VIRVS Data Pair
To make the workshop runnable instantly without downloading multi-gigabyte raw datasets, we use `src.generate_mock_virvs_data` to construct high-fidelity paired micrographs."""),

    code_cell("""import matplotlib.pyplot as plt
from src.generate_mock_virvs_data import create_dataset_directory, generate_virvs_pair

data_dir = "./data/mock_virvs"
create_dataset_directory(data_dir, num_train=30, num_val=10, image_size=(256, 256))

# Visualize a single synthetic sample pair
bf, fluo, masks = generate_virvs_pair(image_size=(256, 256), seed=42)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(bf, cmap="gray")
axes[0].set_title("Input: Brightfield (y)", fontweight="bold")
axes[0].axis("off")

axes[1].imshow(fluo, cmap="magma")
axes[1].set_title("Target: Virus Reporter (x)", fontweight="bold")
axes[1].axis("off")

axes[2].imshow(masks.sum(axis=0), cmap="viridis")
axes[2].set_title("Cell Masks (Ground Truth)", fontweight="bold")
axes[2].axis("off")

plt.tight_layout()
plt.show()"""),

    md_cell("""## 3. PyTorch Data Pipeline
We load the generated dataset into our PyTorch `VIRVSDataset` class and inspect sample tensor batches."""),

    code_cell("""from torch.utils.data import DataLoader
from src.data import VIRVSDataset

# Instantiate PyTorch training & validation datasets
train_dataset = VIRVSDataset(root_dir=data_dir, split="train", normalize_range=(-1.0, 1.0))
val_dataset = VIRVSDataset(root_dir=data_dir, split="val", normalize_range=(-1.0, 1.0))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

batch = next(iter(train_loader))
print("Brightfield tensor shape:", batch["brightfield"].shape)  # [B, 1, H, W]
print("Fluorescence tensor shape:", batch["fluorescence"].shape) # [B, 1, H, W]
print("Min/Max range:", batch["brightfield"].min().item(), batch["brightfield"].max().item())""")
])


# ==============================================================================
# Notebook 01b: Autoencoders, VAEs, and Generative Sampling
# ==============================================================================
nb01b = make_nb([
    md_cell("""# Notebook 01b: Autoencoders, VAEs, and Generative Sampling

### Overview
In this notebook, we explore **Autoencoders (AEs)** and **Variational Autoencoders (VAEs)** for representation learning and generative sampling of cell micrographs.

**Key Learning Objectives**:
1. Train a Convolutional Autoencoder (AE) vs a Variational Autoencoder (VAE).
2. Understand the ELBO loss function ($\text{Reconstruction} + \beta \text{KL}$).
3. Perform **Generative Sampling** by drawing $z \sim \mathcal{N}(0, I)$ to synthesize new cell images from scratch.
4. Perform **Latent Space Interpolation** between a GFP-negative cell ($z_A$) and a GFP-positive cell ($z_B$)."""),

    code_cell("""# ==========================================================
# 1. Environment Setup & Data Loading
# ==========================================================
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if 'google.colab' in sys.modules:
    print("[+] Google Colab detected!")
    !git clone https://github.com/ayakimovich/GenAI4BIA.git /content/GenAI4BIA
    %cd /content/GenAI4BIA/practical
    !pip install -r ../requirements.txt -q
    sys.path.append(os.path.abspath("."))
else:
    sys.path.append(os.path.abspath("."))

from src.data import VIRVSDataset
from src.models import Autoencoder, ConvVAE
from src.generate_mock_virvs_data import create_dataset_directory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Using device: {device}")

data_dir = "./data/mock_virvs"
create_dataset_directory(data_dir, num_train=30, num_val=10, image_size=(256, 256))

train_dataset = VIRVSDataset(root_dir=data_dir, split="train", normalize_range=(0.0, 1.0))
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)"""),

    md_cell("""## 2. Train Variational Autoencoder (VAE)
The VAE maps fluorescence cell micrographs $x$ to latent distribution $(\mu, \sigma)$ and optimizes the ELBO objective."""),

    code_cell("""vae = ConvVAE(in_channels=1, latent_dim=32).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

def vae_loss(x_recon, x, mu, log_var, beta=1.0):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

num_epochs = 10
print("[+] Training Variational Autoencoder (VAE)...")
for epoch in range(1, num_epochs + 1):
    vae.train()
    total_loss_accum = 0.0
    for batch in train_loader:
        x = batch["fluorescence"].to(device)
        optimizer.zero_grad()
        x_recon, mu, log_var = vae(x)
        loss, r_loss, k_loss = vae_loss(x_recon, x, mu, log_var, beta=1.0)
        loss.backward()
        optimizer.step()
        total_loss_accum += loss.item()
        
    if epoch % 2 == 0 or epoch == num_epochs:
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Total Loss: {total_loss_accum / len(train_dataset):.2f}")"""),

    md_cell("""## 3. Generative Sampling: $z \sim \mathcal{N}(0, I)$
Because the KL divergence regularized the latent space against a standard Gaussian prior $\mathcal{N}(0, I)$, we can synthesize brand new cell images by sampling random $z$ vectors!"""),

    code_cell("""# Draw 8 random latent samples from prior N(0, I)
with torch.no_grad():
    synthetic_cells = vae.sample(num_samples=8, device=device)

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    cell_img = synthetic_cells[i, 0].cpu().numpy()
    ax.imshow(cell_img, cmap="magma")
    ax.set_title(f"Sample {i+1}", fontsize=10)
    ax.axis("off")

plt.suptitle("Generative Sampling from VAE Prior: z ~ N(0, I)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()"""),

    md_cell("""## 4. Latent Space Interpolation: GFP-Negative to GFP-Positive
We pick two latent vectors ($z_A$ representing a low-expression GFP cell and $z_B$ representing a high-expression GFP cell) and linearly interpolate $z(\alpha) = (1-\alpha) z_A + \alpha z_B$."""),

    code_cell("""# Pick two samples from validation batch
batch = next(iter(train_loader))
x_batch = batch["fluorescence"].to(device)

with torch.no_grad():
    mu, _ = vae.encode(x_batch)
    zA = mu[0:1] # Cell A latent vector
    zB = mu[1:2] # Cell B latent vector

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
fig, axes = plt.subplots(1, len(alphas), figsize=(12, 3))

with torch.no_grad():
    for idx, alpha in enumerate(alphas):
        z_interp = (1.0 - alpha) * zA + alpha * zB
        cell_interp = vae.decode(z_interp)[0, 0].cpu().numpy()
        axes[idx].imshow(cell_interp, cmap="magma")
        axes[idx].set_title(f"α = {alpha:.2f}", fontsize=10, fontweight="bold")
        axes[idx].axis("off")

plt.suptitle("Latent Space Interpolation (Cell Morphing)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()""")
])


# ==============================================================================
# Notebook 02: Baseline Regression U-Net
# ==============================================================================
nb02 = make_nb([
    md_cell("""# Notebook 02: Baseline U-Net Regression for Virtual Staining

### Overview
In this notebook, we train a 2D **U-Net** regression model to predict continuous fluorescent virus infection reporter intensity from label-free brightfield images.

**Key Learning Objectives**:
1. Define the 2D U-Net architecture with skip connections.
2. Train the model using Mean Absolute Error ($L_1$) loss.
3. Monitor training/validation loss curves.
4. Save model checkpoints and inspect virtual staining outputs."""),

    code_cell("""# ==========================================================
# 1. Google Colab Setup & Environment Setup
# ==========================================================
import sys
import os

if 'google.colab' in sys.modules:
    print("[+] Google Colab detected!")
    !git clone https://github.com/ayakimovich/GenAI4BIA.git /content/GenAI4BIA
    %cd /content/GenAI4BIA/practical
    !pip install -r ../requirements.txt -q
    sys.path.append(os.path.abspath("."))
else:
    sys.path.append(os.path.abspath("."))"""),

    md_cell("""## 2. Load Dataset & Instantiate U-Net"""),

    code_cell("""import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data import VIRVSDataset
from src.models import UNet
from src.generate_mock_virvs_data import create_dataset_directory
from src.utils import plot_virtual_staining_comparison

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Using device: {device}")

# Ensure dataset exists
data_dir = "./data/mock_virvs"
create_dataset_directory(data_dir, num_train=30, num_val=10, image_size=(256, 256))

# U-Net uses range [0, 1]
train_dataset = VIRVSDataset(root_dir=data_dir, split="train", normalize_range=(0.0, 1.0))
val_dataset = VIRVSDataset(root_dir=data_dir, split="val", normalize_range=(0.0, 1.0))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Instantiate U-Net Model
model = UNet(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()  # MAE Loss"""),

    md_cell("""## 3. Training Loop"""),

    code_cell("""num_epochs = 15
train_losses, val_losses = [], []

print("[+] Training U-Net Virtual Staining Baseline...")
for epoch in range(1, num_epochs + 1):
    model.train()
    running_train_loss = 0.0
    
    for batch in train_loader:
        bf = batch["brightfield"].to(device)
        fluo = batch["fluorescence"].to(device)
        
        optimizer.zero_grad()
        pred = model(bf)
        loss = criterion(pred, fluo)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * bf.size(0)
        
    epoch_train_loss = running_train_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)
    
    # Validation Step
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            bf = batch["brightfield"].to(device)
            fluo = batch["fluorescence"].to(device)
            pred = model(bf)
            loss = criterion(pred, fluo)
            running_val_loss += loss.item() * bf.size(0)
            
    epoch_val_loss = running_val_loss / len(val_dataset)
    val_losses.append(epoch_val_loss)
    
    if epoch % 3 == 0 or epoch == num_epochs:
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Train L1: {epoch_train_loss:.4f} | Val L1: {epoch_val_loss:.4f}")

# Save trained U-Net checkpoint
torch.save(model.state_dict(), "./unet_virvs_baseline.pth")
print("[+] Model saved to unet_virvs_baseline.pth")"""),

    md_cell("""## 4. Plot Loss Curves & Visual Inspection"""),

    code_cell("""plt.figure(figsize=(7, 4))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train L1 Loss", linewidth=2)
plt.plot(range(1, num_epochs + 1), val_losses, label="Val L1 Loss", linestyle="--", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.title("U-Net Virtual Staining Training Curve", fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Visual comparison on validation sample
model.eval()
val_batch = next(iter(val_loader))
with torch.no_grad():
    bf_sample = val_batch["brightfield"][0:1].to(device)
    gt_sample = val_batch["fluorescence"][0:1].to(device)
    pred_sample = model(bf_sample)

plot_virtual_staining_comparison(
    bf_sample[0], gt_sample[0], pred_unet=pred_sample[0],
    title="U-Net Virtual Staining Prediction"
)""")
])


# ==============================================================================
# Notebook 03: Pix2Pix Conditional GAN
# ==============================================================================
nb03 = make_nb([
    md_cell("""# Notebook 03: Pix2Pix Conditional GAN for Virtual Staining

### Overview
In this notebook, we train a **Pix2Pix Conditional Generative Adversarial Network (cGAN)** to perform virtual staining of virus infection fluorescence from brightfield micrographs.

**Key Learning Objectives**:
1. Define the PatchGAN Discriminator ($D$) and U-Net Generator ($G$).
2. Combine Adversarial Loss ($L_{\\text{cGAN}}$) with $L_1$ Reconstruction Loss.
3. Understand adversarial training dynamics and discriminator/generator loss tracking.
4. Save generator checkpoints and inspect high-contrast virtual staining outputs."""),

    code_cell("""# ==========================================================
# 1. Google Colab Setup & Environment Setup
# ==========================================================
import sys
import os

if 'google.colab' in sys.modules:
    print("[+] Google Colab detected!")
    !git clone https://github.com/ayakimovich/GenAI4BIA.git /content/GenAI4BIA
    %cd /content/GenAI4BIA/practical
    !pip install -r ../requirements.txt -q
    sys.path.append(os.path.abspath("."))
else:
    sys.path.append(os.path.abspath("."))"""),

    md_cell("""## 2. Initialize Models & Data Loaders"""),

    code_cell("""import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data import VIRVSDataset
from src.models import Pix2PixGenerator, PatchGANDiscriminator
from src.generate_mock_virvs_data import create_dataset_directory
from src.utils import plot_virtual_staining_comparison

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Using device: {device}")

data_dir = "./data/mock_virvs"
create_dataset_directory(data_dir, num_train=30, num_val=10, image_size=(256, 256))

# Pix2Pix GAN uses normalized range [-1, 1]
train_dataset = VIRVSDataset(root_dir=data_dir, split="train", normalize_range=(-1.0, 1.0))
val_dataset = VIRVSDataset(root_dir=data_dir, split="val", normalize_range=(-1.0, 1.0))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Instantiate Generator and Discriminator
netG = Pix2PixGenerator(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
netD = PatchGANDiscriminator(in_channels=2, features=[32, 64, 128, 256]).to(device)

optG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))

criterion_gan = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()
lambda_l1 = 100.0"""),

    md_cell("""## 3. Adversarial Training Loop"""),

    code_cell("""num_epochs = 15
g_losses, d_losses = [], []

print("[+] Training Pix2Pix Conditional GAN...")
for epoch in range(1, num_epochs + 1):
    netG.train()
    netD.train()
    running_g, running_d = 0.0, 0.0
    
    for batch in train_loader:
        y_bf = batch["brightfield"].to(device)
        x_real = batch["fluorescence"].to(device)
        
        # --------------------------------------------------
        # 1. Update Discriminator D
        # --------------------------------------------------
        optD.zero_grad()
        
        pred_real = netD(y_bf, x_real)
        loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
        
        x_fake = netG(y_bf)
        pred_fake = netD(y_bf, x_fake.detach())
        loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
        
        loss_D = (loss_d_real + loss_d_fake) * 0.5
        loss_D.backward()
        optD.step()
        
        # --------------------------------------------------
        # 2. Update Generator G
        # --------------------------------------------------
        optG.zero_grad()
        pred_fake_g = netD(y_bf, x_fake)
        loss_g_gan = criterion_gan(pred_fake_g, torch.ones_like(pred_fake_g))
        loss_g_l1 = criterion_l1(x_fake, x_real)
        
        loss_G = loss_g_gan + lambda_l1 * loss_g_l1
        loss_G.backward()
        optG.step()
        
        running_d += loss_D.item() * y_bf.size(0)
        running_g += loss_G.item() * y_bf.size(0)
        
    epoch_d = running_d / len(train_dataset)
    epoch_g = running_g / len(train_dataset)
    d_losses.append(epoch_d)
    g_losses.append(epoch_g)
    
    if epoch % 3 == 0 or epoch == num_epochs:
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Loss D: {epoch_d:.4f} | Loss G: {epoch_g:.4f}")

# Save trained Pix2Pix generator checkpoint
torch.save(netG.state_dict(), "./pix2pix_generator_virvs.pth")
print("[+] Model saved to pix2pix_generator_virvs.pth")"""),

    md_cell("""## 4. Plot Loss Curves & Visual Inspection"""),

    code_cell("""plt.figure(figsize=(7, 4))
plt.plot(range(1, num_epochs + 1), g_losses, label="Generator Loss", linewidth=2)
plt.plot(range(1, num_epochs + 1), d_losses, label="Discriminator Loss", linestyle="--", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Pix2Pix Conditional GAN Training Dynamics", fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Visual comparison on validation sample
netG.eval()
val_batch = next(iter(val_loader))
with torch.no_grad():
    bf_sample = val_batch["brightfield"][0:1].to(device)
    gt_sample = val_batch["fluorescence"][0:1].to(device)
    pred_sample = netG(bf_sample)
    
    bf_vis = (bf_sample[0] + 1.0) / 2.0
    gt_vis = (gt_sample[0] + 1.0) / 2.0
    pred_vis = (pred_sample[0] + 1.0) / 2.0

plot_virtual_staining_comparison(
    bf_vis, gt_vis, pred_pix2pix=pred_vis,
    title="Pix2Pix Virtual Staining Prediction"
)""")
])


# ==============================================================================
# Notebook 04: Evaluation & VIRVS Benchmarking
# ==============================================================================
nb04 = make_nb([
    md_cell("""# Notebook 04: Evaluation & VIRVS Benchmarking Suite

### Overview
In this notebook, we perform quantitative benchmarking comparing the **U-Net Baseline** and the **Pix2Pix Conditional GAN** models on the VIRVS test set.

**Key Learning Objectives**:
1. Calculate standard image fidelity metrics: **PSNR**, **SSIM**, **PCC**, and **MAE**.
2. Generate a structured quantitative benchmark comparison table.
3. Perform single-cell infection reporter signal quantification ($I_{\\text{viral}}$).
4. Plot ground truth vs predicted viral reporter intensities across cells."""),

    code_cell("""# ==========================================================
# 1. Google Colab Setup & Environment Setup
# ==========================================================
import sys
import os

if 'google.colab' in sys.modules:
    print("[+] Google Colab detected!")
    !git clone https://github.com/ayakimovich/GenAI4BIA.git /content/GenAI4BIA
    %cd /content/GenAI4BIA/practical
    !pip install -r ../requirements.txt -q
    sys.path.append(os.path.abspath("."))
else:
    sys.path.append(os.path.abspath("."))"""),

    md_cell("""## 2. Load Checkpoints & Run Validation Inference"""),

    code_cell("""import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data import VIRVSDataset
from src.models import UNet, Pix2PixGenerator
from src.metrics import compute_psnr, compute_ssim, compute_pcc, compute_mae, compute_cell_reporter_stats
from src.generate_mock_virvs_data import create_dataset_directory, generate_virvs_pair
from src.utils import plot_virtual_staining_comparison

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "./data/mock_virvs"
create_dataset_directory(data_dir, num_train=30, num_val=10, image_size=(256, 256))

# Load validation set in [0, 1] for metric calculation
val_dataset = VIRVSDataset(root_dir=data_dir, split="val", normalize_range=(0.0, 1.0))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Load U-Net & Pix2Pix Checkpoints
unet = UNet(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
if os.path.exists("./unet_virvs_baseline.pth"):
    unet.load_state_dict(torch.load("./unet_virvs_baseline.pth", map_location=device))
unet.eval()

pix2pix = Pix2PixGenerator(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
if os.path.exists("./pix2pix_generator_virvs.pth"):
    pix2pix.load_state_dict(torch.load("./pix2pix_generator_virvs.pth", map_location=device))
pix2pix.eval()"""),

    md_cell("""## 3. Quantitative Image Fidelity Metrics (PSNR, SSIM, PCC, MAE)"""),

    code_cell("""metrics_unet = {"psnr": [], "ssim": [], "pcc": [], "mae": []}
metrics_pix2pix = {"psnr": [], "ssim": [], "pcc": [], "mae": []}

with torch.no_grad():
    for batch in val_loader:
        bf = batch["brightfield"].to(device)       # [1, 1, H, W] in [0, 1]
        target = batch["fluorescence"].to(device)   # [1, 1, H, W] in [0, 1]
        
        # U-Net prediction in [0, 1]
        pred_unet = unet(bf)
        
        # Pix2Pix prediction (convert input to [-1, 1], convert output back to [0, 1])
        bf_pix = bf * 2.0 - 1.0
        pred_pix_raw = pix2pix(bf_pix)
        pred_pix = (pred_pix_raw + 1.0) / 2.0
        
        # Record U-Net metrics
        metrics_unet["psnr"].append(compute_psnr(target, pred_unet))
        metrics_unet["ssim"].append(compute_ssim(target, pred_unet))
        metrics_unet["pcc"].append(compute_pcc(target, pred_unet))
        metrics_unet["mae"].append(compute_mae(target, pred_unet))
        
        # Record Pix2Pix metrics
        metrics_pix2pix["psnr"].append(compute_psnr(target, pred_pix))
        metrics_pix2pix["ssim"].append(compute_ssim(target, pred_pix))
        metrics_pix2pix["pcc"].append(compute_pcc(target, pred_pix))
        metrics_pix2pix["mae"].append(compute_mae(target, pred_pix))

# Create Benchmark Summary Table
summary_data = {
    "Model": ["U-Net Baseline", "Pix2Pix cGAN"],
    "PSNR (dB) ↑": [f"{np.mean(metrics_unet['psnr']):.2f} ± {np.std(metrics_unet['psnr']):.2f}",
                    f"{np.mean(metrics_pix2pix['psnr']):.2f} ± {np.std(metrics_pix2pix['psnr']):.2f}"],
    "SSIM ↑": [f"{np.mean(metrics_unet['ssim']):.3f} ± {np.std(metrics_unet['ssim']):.3f}",
               f"{np.mean(metrics_pix2pix['ssim']):.3f} ± {np.std(metrics_pix2pix['ssim']):.3f}"],
    "PCC ↑": [f"{np.mean(metrics_unet['pcc']):.3f} ± {np.std(metrics_unet['pcc']):.3f}",
              f"{np.mean(metrics_pix2pix['pcc']):.3f} ± {np.std(metrics_pix2pix['pcc']):.3f}"],
    "MAE ↓": [f"{np.mean(metrics_unet['mae']):.4f} ± {np.std(metrics_unet['mae']):.4f}",
              f"{np.mean(metrics_pix2pix['mae']):.4f} ± {np.std(metrics_pix2pix['mae']):.4f}"]
}

df_results = pd.DataFrame(summary_data)
print("\\n================ VIRVS BENCHMARK SUMMARY ================")
print(df_results.to_string(index=False))"""),

    md_cell("""## 4. Single-Cell Infection Reporter Signal Quantification
We evaluate how well predicted virtual stains quantify true single-cell viral infection intensity."""),

    code_cell("""# Generate a test sample with known single-cell masks
bf_arr, fluo_arr, cell_masks = generate_virvs_pair(image_size=(256, 256), seed=999)

bf_t = torch.from_numpy(bf_arr).unsqueeze(0).unsqueeze(0).to(device)
fluo_t = torch.from_numpy(fluo_arr).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    pred_unet_cell = unet(bf_t)[0, 0].cpu().numpy()
    pred_pix_cell = ((pix2pix(bf_t * 2.0 - 1.0)[0, 0] + 1.0) / 2.0).cpu().numpy()

# Compute single-cell viral reporter intensity stats
unet_cell_stats = compute_cell_reporter_stats(fluo_arr, pred_unet_cell, cell_masks)
pix_cell_stats = compute_cell_reporter_stats(fluo_arr, pred_pix_cell, cell_masks)

plt.figure(figsize=(7, 5))
plt.scatter(unet_cell_stats["true_intensities"], unet_cell_stats["pred_intensities"], 
            color="tab:blue", label=f"U-Net (Cell PCC: {unet_cell_stats['cell_pcc']:.2f})", s=50, alpha=0.8)
plt.scatter(pix_cell_stats["true_intensities"], pix_cell_stats["pred_intensities"], 
            color="tab:orange", label=f"Pix2Pix (Cell PCC: {pix_cell_stats['cell_pcc']:.2f})", s=50, alpha=0.8)
plt.plot([0, 1], [0, 1], "k--", label="Ideal 1:1 Line", alpha=0.7)

plt.xlabel("True Single-Cell Viral Reporter Intensity", fontweight="bold")
plt.ylabel("Predicted Virtual Stain Intensity", fontweight="bold")
plt.title("Single-Cell Infection Quantification Fidelity", fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Side-by-Side Visual Comparison
plot_virtual_staining_comparison(
    bf_arr, fluo_arr, pred_unet=pred_unet_cell, pred_pix2pix=pred_pix_cell,
    title="VIRVS Benchmark Comparison: U-Net vs Pix2Pix"
)""")
])


# ==============================================================================
# Master Notebook: Complete Practical Workshop (All Sections Consolidated)
# ==============================================================================
nb_master = make_nb([
    md_cell("""# Practical Workshop: Generative AI for Bioimage Analysis (VIRVS Benchmark)

Welcome to the hands-on practical workshop! This consolidated notebook covers the entire hands-on tutorial series reproducing the **VIRVS** (Virus Infection Reporter Virtual Staining) benchmark from:
> Wyrzykowska, M., della Maggiora, G., Deshpande, N., Mokarian, A., & Yakimovich, A. (2024/2025). *A Benchmark for Virus Infection Reporter Virtual Staining in Fluorescence and Brightfield Microscopy*. Scientific Data / bioRxiv. [GitHub: casus/virvs](https://github.com/casus/virvs)

---

### Workshop Navigation & Structure:
- **Section 1: Data Download & Preparation** (Dataset pair structure, RODARE access, synthetic mock generator)
- **Section 2: Autoencoders, VAEs & Generative Sampling** (ELBO objective, latent sampling $z \\sim \\mathcal{N}(0, I)$, latent space interpolation)
- **Section 3: U-Net Baseline Regression** (Continuous viral reporter fluorescence prediction)
- **Section 4: Pix2Pix Conditional GAN** (Generative virtual staining model with adversarial loss)
- **Section 5: Evaluation & Benchmarking** (Image fidelity PSNR, SSIM, PCC, MAE & single-cell viral quantification)

*Note: All data and trained checkpoints persist in memory/disk across sections within this single Colab session!*"""),

    code_cell("""# ==========================================================
# 0. Google Colab Setup & Environment Initialization
# ==========================================================
import sys
import os

if 'google.colab' in sys.modules:
    print("[+] Google Colab detected! Installing dependencies and setting up paths...")
    !git clone https://github.com/ayakimovich/GenAI4BIA.git /content/GenAI4BIA
    %cd /content/GenAI4BIA/practical
    !pip install -r ../requirements.txt -q
    sys.path.append(os.path.abspath("."))
else:
    print("[+] Local execution detected.")
    sys.path.append(os.path.abspath("."))"""),

    # --- SECTION 1 ---
    md_cell("""# Section 1: Data Download & Preparation (VIRVS Benchmark)

In this section, we explore the dataset structure introduced in the VIRVS benchmark. The benchmark maps non-destructive label-free **Brightfield** micrographs ($y$) to continuous **Fluorescence** infection reporter signals ($x$).

**Morphological Inductive Bias**:
To provide a clear, learnable mapping in synthetic data, we enforce a strict morphological rule:
- **Infected Cells**: Rendered in Brightfield as **circles with a central hole** (hollow/donut shape) $\\rightarrow$ express **positive fluorescence**.
- **Uninfected Cells**: Rendered in Brightfield as **solid circles without a hole** $\\rightarrow$ express **zero fluorescence**.

**Key Learning Objectives**:
1. Understand dataset pair structure for virtual staining.
2. Generate/load paired micrographs enforcing morphological inductive bias.
3. Implement PyTorch `Dataset` & `DataLoader` pipelines."""),

    code_cell("""import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.generate_mock_virvs_data import create_dataset_directory, generate_virvs_pair
from src.data import VIRVSDataset

data_dir = "./data/mock_virvs"
create_dataset_directory(data_dir, num_train=30, num_val=10, image_size=(256, 256))

# Visualize a single synthetic sample pair
bf, fluo, masks = generate_virvs_pair(image_size=(256, 256), seed=42)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(bf, cmap="gray")
axes[0].set_title("Input: Brightfield (y)", fontweight="bold")
axes[0].axis("off")

axes[1].imshow(fluo, cmap="magma")
axes[1].set_title("Target: Virus Reporter (x)", fontweight="bold")
axes[1].axis("off")

axes[2].imshow(masks.sum(axis=0), cmap="viridis")
axes[2].set_title("Cell Masks (Ground Truth)", fontweight="bold")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# Instantiate PyTorch training & validation datasets
train_dataset = VIRVSDataset(root_dir=data_dir, split="train", normalize_range=(-1.0, 1.0))
val_dataset = VIRVSDataset(root_dir=data_dir, split="val", normalize_range=(-1.0, 1.0))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

batch = next(iter(train_loader))
print("Brightfield tensor shape:", batch["brightfield"].shape)  # [B, 1, H, W]
print("Fluorescence tensor shape:", batch["fluorescence"].shape) # [B, 1, H, W]
print("Min/Max range:", batch["brightfield"].min().item(), batch["brightfield"].max().item())"""),

    # --- SECTION 2 ---
    md_cell("""# Section 2: Autoencoders, VAEs & Generative Sampling

In this section, we explore **Autoencoders (AEs)** and **Variational Autoencoders (VAEs)** for representation learning and generative sampling of cell micrographs.

**Key Learning Objectives**:
1. Train a Convolutional Variational Autoencoder (VAE).
2. Understand the ELBO loss function ($\\text{Reconstruction} + \\beta \\text{KL}$).
3. Perform **Generative Sampling** by drawing $z \\sim \\mathcal{N}(0, I)$ to synthesize new cell images.
4. Perform **Latent Space Interpolation** between a GFP-negative cell ($z_A$) and a GFP-positive cell ($z_B$)."""),

    code_cell("""import torch
import torch.nn as nn
import torch.optim as optim
from src.models import Autoencoder, ConvVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Using device: {device}")

# Dataset normalized in [0, 1] for VAE reconstruction
train_dataset_vae = VIRVSDataset(root_dir=data_dir, split="train", normalize_range=(0.0, 1.0))
train_loader_vae = DataLoader(train_dataset_vae, batch_size=4, shuffle=True)

vae = ConvVAE(in_channels=1, latent_dim=32).to(device)
optimizer_vae = optim.Adam(vae.parameters(), lr=1e-3)

def vae_loss(x_recon, x, mu, log_var, beta=1.0):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

num_epochs_vae = 10
print("[+] Training Variational Autoencoder (VAE)...")
for epoch in range(1, num_epochs_vae + 1):
    vae.train()
    total_loss_accum = 0.0
    for batch in train_loader_vae:
        x = batch["fluorescence"].to(device)
        optimizer_vae.zero_grad()
        x_recon, mu, log_var = vae(x)
        loss, r_loss, k_loss = vae_loss(x_recon, x, mu, log_var, beta=1.0)
        loss.backward()
        optimizer_vae.step()
        total_loss_accum += loss.item()
        
    if epoch % 2 == 0 or epoch == num_epochs_vae:
        print(f"Epoch [{epoch:02d}/{num_epochs_vae:02d}] - Total Loss: {total_loss_accum / len(train_dataset_vae):.2f}")

# Generative Sampling z ~ N(0, I)
with torch.no_grad():
    synthetic_cells = vae.sample(num_samples=8, device=device)

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    cell_img = synthetic_cells[i, 0].cpu().numpy()
    ax.imshow(cell_img, cmap="magma")
    ax.set_title(f"Sample {i+1}", fontsize=10)
    ax.axis("off")

plt.suptitle("Generative Sampling from VAE Prior: z ~ N(0, I)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# Latent Space Interpolation
batch = next(iter(train_loader_vae))
x_batch = batch["fluorescence"].to(device)

with torch.no_grad():
    mu, _ = vae.encode(x_batch)
    zA = mu[0:1]
    zB = mu[1:2]

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
fig, axes = plt.subplots(1, len(alphas), figsize=(12, 3))

with torch.no_grad():
    for idx, alpha in enumerate(alphas):
        z_interp = (1.0 - alpha) * zA + alpha * zB
        cell_interp = vae.decode(z_interp)[0, 0].cpu().numpy()
        axes[idx].imshow(cell_interp, cmap="magma")
        axes[idx].set_title(f"α = {alpha:.2f}", fontsize=10, fontweight="bold")
        axes[idx].axis("off")

plt.suptitle("Latent Space Interpolation (Cell Morphing)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()"""),

    # --- SECTION 3 ---
    md_cell("""# Section 3: Baseline U-Net Regression for Virtual Staining

In this section, we train a 2D **U-Net** regression model to predict continuous fluorescent virus infection reporter intensity directly from label-free brightfield micrographs.

**Key Learning Objectives**:
1. Define the 2D U-Net architecture with skip connections.
2. Train the model using Mean Absolute Error ($L_1$) loss.
3. Monitor training/validation loss curves.
4. Save model checkpoints and inspect virtual staining outputs."""),

    code_cell("""from src.models import UNet
from src.utils import plot_virtual_staining_comparison

train_dataset_unet = VIRVSDataset(root_dir=data_dir, split="train", normalize_range=(0.0, 1.0))
val_dataset_unet = VIRVSDataset(root_dir=data_dir, split="val", normalize_range=(0.0, 1.0))

train_loader_unet = DataLoader(train_dataset_unet, batch_size=4, shuffle=True)
val_loader_unet = DataLoader(val_dataset_unet, batch_size=4, shuffle=False)

unet_model = UNet(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
optimizer_unet = optim.Adam(unet_model.parameters(), lr=1e-3)
criterion_unet = nn.L1Loss()

num_epochs_unet = 15
train_losses_unet, val_losses_unet = [], []

print("[+] Training U-Net Virtual Staining Baseline...")
for epoch in range(1, num_epochs_unet + 1):
    unet_model.train()
    running_train_loss = 0.0
    
    for batch in train_loader_unet:
        bf = batch["brightfield"].to(device)
        fluo = batch["fluorescence"].to(device)
        
        optimizer_unet.zero_grad()
        pred = unet_model(bf)
        loss = criterion_unet(pred, fluo)
        loss.backward()
        optimizer_unet.step()
        
        running_train_loss += loss.item() * bf.size(0)
        
    epoch_train_loss = running_train_loss / len(train_dataset_unet)
    train_losses_unet.append(epoch_train_loss)
    
    unet_model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader_unet:
            bf = batch["brightfield"].to(device)
            fluo = batch["fluorescence"].to(device)
            pred = unet_model(bf)
            loss = criterion_unet(pred, fluo)
            running_val_loss += loss.item() * bf.size(0)
            
    epoch_val_loss = running_val_loss / len(val_dataset_unet)
    val_losses_unet.append(epoch_val_loss)
    
    if epoch % 3 == 0 or epoch == num_epochs_unet:
        print(f"Epoch [{epoch:02d}/{num_epochs_unet:02d}] - Train L1: {epoch_train_loss:.4f} | Val L1: {epoch_val_loss:.4f}")

torch.save(unet_model.state_dict(), "./unet_virvs_baseline.pth")
print("[+] Model saved to unet_virvs_baseline.pth")

plt.figure(figsize=(7, 4))
plt.plot(range(1, num_epochs_unet + 1), train_losses_unet, label="Train L1 Loss", linewidth=2)
plt.plot(range(1, num_epochs_unet + 1), val_losses_unet, label="Val L1 Loss", linestyle="--", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.title("U-Net Virtual Staining Training Curve", fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

unet_model.eval()
val_batch = next(iter(val_loader_unet))
with torch.no_grad():
    bf_sample = val_batch["brightfield"][0:1].to(device)
    gt_sample = val_batch["fluorescence"][0:1].to(device)
    pred_sample = unet_model(bf_sample)

plot_virtual_staining_comparison(
    bf_sample[0], gt_sample[0], pred_unet=pred_sample[0],
    title="U-Net Virtual Staining Prediction"
)"""),

    # --- SECTION 4 ---
    md_cell("""# Section 4: Pix2Pix Conditional GAN for Virtual Staining

In this section, we train a **Pix2Pix Conditional Generative Adversarial Network (cGAN)** to perform high-contrast virtual staining of virus infection fluorescence.

**Key Learning Objectives**:
1. Define the PatchGAN Discriminator ($D$) and U-Net Generator ($G$).
2. Combine Adversarial Loss ($L_{\\text{cGAN}}$) with $L_1$ Reconstruction Loss.
3. Understand adversarial training dynamics and discriminator/generator loss tracking.
4. Save generator checkpoints and inspect virtual staining outputs."""),

    code_cell("""from src.models import Pix2PixGenerator, PatchGANDiscriminator

train_dataset_gan = VIRVSDataset(root_dir=data_dir, split="train", normalize_range=(-1.0, 1.0))
val_dataset_gan = VIRVSDataset(root_dir=data_dir, split="val", normalize_range=(-1.0, 1.0))

train_loader_gan = DataLoader(train_dataset_gan, batch_size=4, shuffle=True)
val_loader_gan = DataLoader(val_dataset_gan, batch_size=4, shuffle=False)

netG = Pix2PixGenerator(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
netD = PatchGANDiscriminator(in_channels=2, features=[32, 64, 128, 256]).to(device)

optG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))

criterion_gan = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()
lambda_l1 = 100.0

num_epochs_gan = 15
g_losses, d_losses = [], []

print("[+] Training Pix2Pix Conditional GAN...")
for epoch in range(1, num_epochs_gan + 1):
    netG.train()
    netD.train()
    running_g, running_d = 0.0, 0.0
    
    for batch in train_loader_gan:
        y_bf = batch["brightfield"].to(device)
        x_real = batch["fluorescence"].to(device)
        
        # 1. Update Discriminator D
        optD.zero_grad()
        pred_real = netD(y_bf, x_real)
        loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
        
        x_fake = netG(y_bf)
        pred_fake = netD(y_bf, x_fake.detach())
        loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
        
        loss_D = (loss_d_real + loss_d_fake) * 0.5
        loss_D.backward()
        optD.step()
        
        # 2. Update Generator G
        optG.zero_grad()
        pred_fake_g = netD(y_bf, x_fake)
        loss_g_gan = criterion_gan(pred_fake_g, torch.ones_like(pred_fake_g))
        loss_g_l1 = criterion_l1(x_fake, x_real)
        
        loss_G = loss_g_gan + lambda_l1 * loss_g_l1
        loss_G.backward()
        optG.step()
        
        running_d += loss_D.item() * y_bf.size(0)
        running_g += loss_G.item() * y_bf.size(0)
        
    epoch_d = running_d / len(train_dataset_gan)
    epoch_g = running_g / len(train_dataset_gan)
    d_losses.append(epoch_d)
    g_losses.append(epoch_g)
    
    if epoch % 3 == 0 or epoch == num_epochs_gan:
        print(f"Epoch [{epoch:02d}/{num_epochs_gan:02d}] - Loss D: {epoch_d:.4f} | Loss G: {epoch_g:.4f}")

torch.save(netG.state_dict(), "./pix2pix_generator_virvs.pth")
print("[+] Model saved to pix2pix_generator_virvs.pth")

plt.figure(figsize=(7, 4))
plt.plot(range(1, num_epochs_gan + 1), g_losses, label="Generator Loss", linewidth=2)
plt.plot(range(1, num_epochs_gan + 1), d_losses, label="Discriminator Loss", linestyle="--", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Pix2Pix Conditional GAN Training Dynamics", fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

netG.eval()
val_batch = next(iter(val_loader_gan))
with torch.no_grad():
    bf_sample = val_batch["brightfield"][0:1].to(device)
    gt_sample = val_batch["fluorescence"][0:1].to(device)
    pred_sample = netG(bf_sample)
    
    bf_vis = (bf_sample[0] + 1.0) / 2.0
    gt_vis = (gt_sample[0] + 1.0) / 2.0
    pred_vis = (pred_sample[0] + 1.0) / 2.0

plot_virtual_staining_comparison(
    bf_vis, gt_vis, pred_pix2pix=pred_vis,
    title="Pix2Pix Virtual Staining Prediction"
)"""),

    # --- SECTION 5 ---
    md_cell("""# Section 5: Evaluation & VIRVS Benchmarking Suite

In this section, we perform quantitative benchmarking comparing the **U-Net Baseline** and the **Pix2Pix Conditional GAN** models on the VIRVS test set.

**Key Learning Objectives**:
1. Calculate standard image fidelity metrics: **PSNR**, **SSIM**, **PCC**, and **MAE**.
2. Generate a structured quantitative benchmark comparison table.
3. Perform single-cell infection reporter signal quantification ($I_{\\text{viral}}$).
4. Plot ground truth vs predicted viral reporter intensities across cells."""),

    code_cell("""import numpy as np
import pandas as pd
from src.metrics import compute_psnr, compute_ssim, compute_pcc, compute_mae, compute_cell_reporter_stats

val_dataset_eval = VIRVSDataset(root_dir=data_dir, split="val", normalize_range=(0.0, 1.0))
val_loader_eval = DataLoader(val_dataset_eval, batch_size=1, shuffle=False)

unet_eval = UNet(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
if os.path.exists("./unet_virvs_baseline.pth"):
    unet_eval.load_state_dict(torch.load("./unet_virvs_baseline.pth", map_location=device))
else:
    unet_eval = unet_model
unet_eval.eval()

pix2pix_eval = Pix2PixGenerator(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
if os.path.exists("./pix2pix_generator_virvs.pth"):
    pix2pix_eval.load_state_dict(torch.load("./pix2pix_generator_virvs.pth", map_location=device))
else:
    pix2pix_eval = netG
pix2pix_eval.eval()

metrics_unet = {"psnr": [], "ssim": [], "pcc": [], "mae": []}
metrics_pix2pix = {"psnr": [], "ssim": [], "pcc": [], "mae": []}

with torch.no_grad():
    for batch in val_loader_eval:
        bf = batch["brightfield"].to(device)
        target = batch["fluorescence"].to(device)
        
        pred_unet = unet_eval(bf)
        
        bf_pix = bf * 2.0 - 1.0
        pred_pix_raw = pix2pix_eval(bf_pix)
        pred_pix = (pred_pix_raw + 1.0) / 2.0
        
        metrics_unet["psnr"].append(compute_psnr(target, pred_unet))
        metrics_unet["ssim"].append(compute_ssim(target, pred_unet))
        metrics_unet["pcc"].append(compute_pcc(target, pred_unet))
        metrics_unet["mae"].append(compute_mae(target, pred_unet))
        
        metrics_pix2pix["psnr"].append(compute_psnr(target, pred_pix))
        metrics_pix2pix["ssim"].append(compute_ssim(target, pred_pix))
        metrics_pix2pix["pcc"].append(compute_pcc(target, pred_pix))
        metrics_pix2pix["mae"].append(compute_mae(target, pred_pix))

summary_data = {
    "Model": ["U-Net Baseline", "Pix2Pix cGAN"],
    "PSNR (dB) ↑": [f"{np.mean(metrics_unet['psnr']):.2f} ± {np.std(metrics_unet['psnr']):.2f}",
                    f"{np.mean(metrics_pix2pix['psnr']):.2f} ± {np.std(metrics_pix2pix['psnr']):.2f}"],
    "SSIM ↑": [f"{np.mean(metrics_unet['ssim']):.3f} ± {np.std(metrics_unet['ssim']):.3f}",
               f"{np.mean(metrics_pix2pix['ssim']):.3f} ± {np.std(metrics_pix2pix['ssim']):.3f}"],
    "PCC ↑": [f"{np.mean(metrics_unet['pcc']):.3f} ± {np.std(metrics_unet['pcc']):.3f}",
              f"{np.mean(metrics_pix2pix['pcc']):.3f} ± {np.std(metrics_pix2pix['pcc']):.3f}"],
    "MAE ↓": [f"{np.mean(metrics_unet['mae']):.4f} ± {np.std(metrics_unet['mae']):.4f}",
              f"{np.mean(metrics_pix2pix['mae']):.4f} ± {np.std(metrics_pix2pix['mae']):.4f}"]
}

df_results = pd.DataFrame(summary_data)
print("\\n================ VIRVS BENCHMARK SUMMARY ================")
print(df_results.to_string(index=False))

# Single-Cell Infection Quantification
bf_arr, fluo_arr, cell_masks = generate_virvs_pair(image_size=(256, 256), seed=999)
bf_t = torch.from_numpy(bf_arr).unsqueeze(0).unsqueeze(0).to(device)
fluo_t = torch.from_numpy(fluo_arr).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    pred_unet_cell = unet_eval(bf_t)[0, 0].cpu().numpy()
    pred_pix_cell = ((pix2pix_eval(bf_t * 2.0 - 1.0)[0, 0] + 1.0) / 2.0).cpu().numpy()

unet_cell_stats = compute_cell_reporter_stats(fluo_arr, pred_unet_cell, cell_masks)
pix_cell_stats = compute_cell_reporter_stats(fluo_arr, pred_pix_cell, cell_masks)

plt.figure(figsize=(7, 5))
plt.scatter(unet_cell_stats["true_intensities"], unet_cell_stats["pred_intensities"], 
            color="tab:blue", label=f"U-Net (Cell PCC: {unet_cell_stats['cell_pcc']:.2f})", s=50, alpha=0.8)
plt.scatter(pix_cell_stats["true_intensities"], pix_cell_stats["pred_intensities"], 
            color="tab:orange", label=f"Pix2Pix (Cell PCC: {pix_cell_stats['cell_pcc']:.2f})", s=50, alpha=0.8)
plt.plot([0, 1], [0, 1], "k--", label="Ideal 1:1 Line", alpha=0.7)

plt.xlabel("True Single-Cell Viral Reporter Intensity", fontweight="bold")
plt.ylabel("Predicted Virtual Stain Intensity", fontweight="bold")
plt.title("Single-Cell Infection Quantification Fidelity", fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Side-by-Side Visual Comparison
plot_virtual_staining_comparison(
    bf_arr, fluo_arr, pred_unet=pred_unet_cell, pred_pix2pix=pred_pix_cell,
    title="VIRVS Benchmark Comparison: U-Net vs Pix2Pix"
)""")
])


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(script_dir, "practical_workshop.ipynb"), "w") as f:
        json.dump(nb_master, f, indent=1)
    print("[+] Created practical_workshop.ipynb (Consolidated Master Notebook)")


if __name__ == "__main__":
    main()

