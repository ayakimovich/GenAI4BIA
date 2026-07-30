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
   - 🚀 **[Practical Workshop Notebook (`practical_workshop.ipynb`)](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/practical_workshop.ipynb)**: Complete hands-on tutorial divided into 5 interactive sections:
     1. **Section 1**: Data Download & Preparation (VIRVS dataset layout, RODARE access, synthetic mock generator).
     2. **Section 2**: Autoencoders & VAEs (ELBO objective, generative sampling $z \sim \mathcal{N}(0, I)$, latent space interpolation).
     3. **Section 3**: U-Net Baseline Regression (Predicting continuous infection fluorescence from brightfield).
     4. **Section 4**: Pix2Pix Conditional GAN (Generative virtual staining model with adversarial loss).
     5. **Section 5**: Evaluation & Benchmarking (PSNR, SSIM, PCC, MAE, cell-level viral reporter signal quantification).

---

## Schedule (Half-Day Workshop)

| Time          | Session                       | Description                                                                                                                                                                                          |
| :--------------| :------------------------------| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 09:00 - 10:30 | **Lecture (Slides)**          | Inverse problems, distribution learning, VAEs, Pix2Pix GANs, Diffusion models                                                                                                                       |
| 10:30 - 10:45 | **Coffee Break**              |                                                                                                                                                                                                      |
| 10:45 - 12:15 | **Hands-on Workshop**         | [Practical Workshop Notebook (Sections 1–4)](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/practical_workshop.ipynb): Data loading, VAEs, U-Net regression, Pix2Pix GAN training |
| 12:15 - 12:45 | **Benchmarking & Discussion** | [Practical Workshop Notebook (Section 5)](https://colab.research.google.com/github/ayakimovich/GenAI4BIA/blob/main/practical/practical_workshop.ipynb): Metric evaluation (PSNR, SSIM, PCC, cell reporter stats) & VIRVS benchmarks  |
| 12:45 - 13:00 | **Q&A & Wrap-up**             | Future directions: Diffusion models, physics-informed priors, unpaired translation                                                                                                                  |

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
