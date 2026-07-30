# Practical Workshop: VIRVS Benchmark Principles

Hands-on tutorial series creating synthetic data to illustrate the principles of the **VIRVS** (Virus Infection Reporter Virtual Staining) benchmark from:
> Wyrzykowska, M., della Maggiora, G., Deshpande, N., Mokarian, A., & Yakimovich, A. (2024/2025). *A Benchmark for Virus Infection Reporter Virtual Staining in Fluorescence and Brightfield Microscopy*. Scientific Data / bioRxiv. [GitHub: casus/virvs](https://github.com/casus/virvs)

---

## Workshop Notebook

🚀 **[Practical Workshop Notebook (`practical_workshop.ipynb`)](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/practical_workshop.ipynb)**: Recommended for Google Colab. Runs the entire 5-part workshop within a single active session so generated data and trained model checkpoints persist seamlessly across sections:

1. **Section 1: Data Download & Preparation** (Dataset pair structure, RODARE access, synthetic mock generator)
2. **Section 2: Autoencoders, VAEs & Generative Sampling** (ELBO objective, latent sampling $z \sim \mathcal{N}(0, I)$, latent space interpolation)
3. **Section 3: Baseline U-Net Regression for Virtual Staining** (Continuous viral reporter fluorescence prediction)
4. **Section 4: Pix2Pix Conditional GAN for Virtual Staining** (Generative virtual staining model with adversarial loss)
5. **Section 5: Evaluation & VIRVS Benchmarking Suite** (Image fidelity PSNR, SSIM, PCC, MAE & single-cell viral quantification)

---

## Shared Source Package (`src/`)

- `src/generate_mock_virvs_data.py`: Synthetic paired Brightfield vs Infection Reporter dataset generator.
- `src/data.py`: `VIRVSDataset` PyTorch Dataset class.
- `src/models.py`: `UNet`, `Autoencoder`, `ConvVAE`, `Pix2PixGenerator`, `PatchGANDiscriminator` architectures.
- `src/metrics.py`: `compute_psnr`, `compute_ssim`, `compute_pcc`, `compute_mae`, `compute_cell_reporter_stats`.
- `src/utils.py`: Visualization helpers for model predictions and difference maps.

---

## Quickstart

To run the complete workshop in a single session:

```bash
jupyter notebook practical_workshop.ipynb
```
