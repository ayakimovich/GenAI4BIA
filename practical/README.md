# Practical Workshop: VIRVS Benchmark Reproduction

Hands-on tutorial series reproducing the **VIRVS** (Virus Infection Reporter Virtual Staining) benchmark from:
> Wyrzykowska, M., della Maggiora, G., Deshpande, N., Mokarian, A., & Yakimovich, A. (2024/2025). *A Benchmark for Virus Infection Reporter Virtual Staining in Fluorescence and Brightfield Microscopy*. Scientific Data / bioRxiv. [GitHub: casus/virvs](https://github.com/casus/virvs)

---

## Notebook Structure

| Notebook | Topic | Description |
| :--- | :--- | :--- |
| [`01_data_download_and_prep.ipynb`](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/01_data_download_and_prep.ipynb) | **Data Pipeline** | Loading VIRVS datasets, dataset structure, mock generator for quick testing, normalization |
| [`01b_autoencoders_and_vaes.ipynb`](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/01b_autoencoders_and_vaes.ipynb) | **AEs, VAEs & Generative Sampling** | Autoencoders vs VAEs, ELBO objective, generative sampling $z \sim \mathcal{N}(0, I)$, latent space interpolation |
| [`02_baseline_regression_unet.ipynb`](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/02_baseline_regression_unet.ipynb) | **U-Net Baseline** | Training 2D U-Net for continuous viral reporter fluorescence prediction ($L_1 / L_2$ loss) |
| [`03_generative_pix2pix_gan.ipynb`](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/03_generative_pix2pix_gan.ipynb) | **Pix2Pix cGAN** | Training conditional GAN virtual staining model ($L_{\text{cGAN}} + \lambda L_1$) |
| [`04_evaluation_and_benchmarking.ipynb`](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/04_evaluation_and_benchmarking.ipynb) | **VIRVS Benchmarking** | Model evaluation across PSNR, SSIM, PCC, MAE, and cell-level viral reporter quantification |

---

## Shared Source Package (`src/`)

- `src/generate_mock_virvs_data.py`: Synthetic paired Brightfield vs Infection Reporter dataset generator.
- `src/data.py`: `VIRVSDataset` PyTorch Dataset class.
- `src/models.py`: `UNet`, `Autoencoder`, `ConvVAE`, `Pix2PixGenerator`, `PatchGANDiscriminator` architectures.
- `src/metrics.py`: `compute_psnr`, `compute_ssim`, `compute_pcc`, `compute_mae`, `compute_cell_reporter_stats`.
- `src/utils.py`: Visualization helpers for model predictions and difference maps.

---

## Quickstart

Run [Notebook 01](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/01_data_download_and_prep.ipynb) first to verify environment setup and generate initial training data:

```bash
jupyter notebook 01_data_download_and_prep.ipynb
```
