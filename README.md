# Generative AI for Bioimage Analysis

A half-day course covering the mathematical foundations of generative AI, inverse problems, and distribution learning in bioimage analysis, paired with a practical hands-on benchmark reproduction of **VIRVS** (Virus Infection Reporter Virtual Staining).

---

## Course Overview

This repository is divided into two primary components:

1. **`slides/` — Theoretical Foundations (LaTeX Slide Deck)**
   - **Inverse Problems in Bioimaging**: Forward model $y = Ax + n$, ill-posedness, culminating in **Virtual Staining** (mapping label-free brightfield micrographs to continuous fluorescence infection reporter signals).
   - **Distribution Learning**: Empirical distributions $p_{\text{data}}(x)$, Maximum Likelihood Estimation (MLE), KL divergence, Jensen-Shannon divergence, Wasserstein distance.
   - **Generative Model Families**:
     - *Variational Autoencoders (VAEs)*: ELBO objective, reparameterization trick $z = \mu + \sigma \odot \epsilon$.
     - *Generative Adversarial Networks (GANs / Pix2Pix)*: Minimax objective $\min_G \max_D V(D,G)$, conditional GAN loss + L1 reconstruction.
     - *Diffusion Models (DDPM)*: Forward noise addition $q(x_t|x_{t-1})$, reverse process $p_\theta(x_{t-1}|x_t)$, noise prediction MSE loss $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$.
   - **Every mathematical equation is presented alongside its Python / PyTorch code equivalent!**

2. **`practical/` — VIRVS Benchmark Reproduction (Hands-on Tutorials)**
   - Based on the paper:
     > Wyrzykowska, M., della Maggiora, G., Deshpande, N., Mokarian, A., & Yakimovich, A. (2024/2025). *A Benchmark for Virus Infection Reporter Virtual Staining in Fluorescence and Brightfield Microscopy*. Scientific Data / bioRxiv. [GitHub: casus/virvs](https://github.com/casus/virvs)
   - [Notebook 01](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/01_data_download_and_prep.ipynb): Data Download & Preparation (VIRVS dataset layout, RODARE access, synthetic mock generator).
   - [Notebook 01b](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/01b_autoencoders_and_vaes.ipynb): Autoencoders & VAEs (Autoencoders vs VAEs, ELBO objective, generative sampling, latent space interpolation).
   - [Notebook 02](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/02_baseline_regression_unet.ipynb): U-Net Baseline Regression (Predicting continuous infection fluorescence from brightfield).
   - [Notebook 03](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/03_generative_pix2pix_gan.ipynb): Pix2Pix Conditional GAN (Generative virtual staining model).
   - [Notebook 04](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/04_evaluation_and_benchmarking.ipynb): Evaluation & Benchmarking (PSNR, SSIM, Pearson Correlation Coefficient, MAE, cell-level viral reporter signal quantification).

---

## Schedule (Half-Day Workshop)

| Time          | Session                       | Description                                                                                                                                                                                                                                                                           |
| :--------------| :------------------------------| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 09:00 - 10:30 | **Lecture (Slides)**          | Inverse problems, distribution learning, VAEs, Pix2Pix GANs, Diffusion models                                                                                                                                                                                                         |
| 10:30 - 10:45 | **Coffee Break**              |                                                                                                                                                                                                                                                                                       |
| 10:45 - 12:15 | **Hands-on Workshop**         | [Notebooks 01](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/01_data_download_and_prep.ipynb)–[03](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/03_generative_pix2pix_gan.ipynb): Data loading, U-Net regression, Pix2Pix GAN training |
| 12:15 - 12:45 | **Benchmarking & Discussion** | [Notebook 04](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/04_evaluation_and_benchmarking.ipynb): Metric evaluation (PSNR, SSIM, PCC, cell reporter stats) & VIRVS benchmarks                                                                                 |
| 12:45 - 13:00 | **Q&A & Wrap-up**             | Future directions: Diffusion models, physics-informed priors, unpaired translation                                                                                                                                                                                                    |

---

## Getting Started

### Environment Setup

#### Option A: Using Conda
```bash
conda env create -f environment.yml
conda activate genai-bia-course
```

#### Option B: Using Pip
```bash
pip install -r requirements.txt
```

#### Option C: Google Colab
All notebooks under `practical/` contain one-click Google Colab badges and automated environment installation cells at the top.

---

## Building the Slides

To compile the LaTeX Beamer slide deck into PDF (`slides/genai_bia_course.pdf`):

```bash
cd slides
bash build.sh
```

*(Requires `pdflatex` or `latexmk` installed locally, e.g., via TeX Live / MacTeX).*

---

## Citation & References

If you use these materials or the VIRVS benchmark, please cite:

```bibtex
@article{wyrzykowska2024benchmark,
  title={A Benchmark for Virus Infection Reporter Virtual Staining in Fluorescence and Brightfield Microscopy},
  author={Wyrzykowska, Maria and della Maggiora, Gabriel and Deshpande, Nikita and Mokarian, Ashkan and Yakimovich, Artur},
  journal={bioRxiv},
  pages={2024--08},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

---

## License
Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE).
