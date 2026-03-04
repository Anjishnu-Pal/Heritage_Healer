# Heritage Healer

A deep learning based GenAI project that restores old, scratched, discolored, and damaged photographs into clean, full-color images. Three independent model architectures are implemented and compared: a **Diffusion Model (DDPM)**, a **GAN (pix2pix)**, and a **VAE (Variational Autoencoder)**.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup & Installation](#setup--installation)
- [Dataset Preparation](#dataset-preparation)
- [Running the Scripts](#running-the-scripts)
- [Outputs](#outputs)
- [Key Hyperparameters](#key-hyperparameters)

---

## Overview

| Component | Diffusion Model | GAN (pix2pix) | VAE |
|---|---|---|---|
| Framework | PyTorch | TensorFlow/Keras | TensorFlow/Keras |
| Architecture | Conditional UNet2D (DDPM) | U-Net Generator + PatchGAN Discriminator | U-Net Encoder–Decoder + Reparameterization |
| Input | 256×256 grayscale (3-ch) | 256×256 grayscale (3-ch) | 256×256 grayscale (1-ch) |
| Output | 256×256 color RGB | 256×256 color RGB | 256×256 color RGB |
| Loss | MSE (noise prediction) | Adversarial + L1 (λ=100) | L1 reconstruction + KL divergence |
| Optimizer | AdamW (lr=1e-4) | Adam (lr=2e-4, β₁=0.5) | Adam (lr=2e-4, β₁=0.5) |
| Epochs | 10 | 10 | 10 |
| Dataset | CelebA (5,000 images) | CelebA (5,000 images) | CelebA (5,000 images) |

---

## Pipeline

```
CelebA Images
      │
      ▼
┌──────────────────────────────┐
│  prepare_dataset.py          │
│  (Diffusion / GANs / VAEs)   │
│  - Resize to 256×256         │
│  - Add random scratches      │
│  - Add Gaussian noise        │
│  - Convert to grayscale      │  ← simulates old/damaged photo
│  - Save paired A / B folders │
└─────────────┬────────────────┘
              │
      ┌───────┼───────────┐
      ▼       ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│Diffusion │ │   GAN    │ │   VAE    │
│ train.py │ │ train.py │ │ train.py │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │             │             │
     ▼             ▼             ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│inference │ │inference │ │inference │
│   .py    │ │   .py    │ │   .py    │
└──────────┘ └──────────┘ └──────────┘
      │             │             │
      └───────┬─────┘─────────────┘
              ▼
   Restored Color Photo
```

---

## Project Structure

```
Heritage_Healer/
│
├── src/
│   ├── Diffusion model/
│   │   ├── prepare_dataset.py   # Download CelebA & build paired A/B dataset
│   │   ├── train.py             # Train conditional DDPM (PyTorch + diffusers)
│   │   └── inference.py         # Restore a single image using saved weights
│   │
│   ├── GANs/
│   │   ├── prepare_dataset.py   # Download CelebA & build paired A/B dataset
│   │   ├── train.py             # Train pix2pix GAN (TensorFlow/Keras)
│   │   └── inference.py         # Restore a single image using saved checkpoint
│   │
│   └── VAEs/
│       ├── train.py             # Download data + train VAE (TensorFlow/Keras)
│       └── inference.py         # Restore a single image using saved model6.h5
│
├── data/
│   └── README.md               # Kaggle API setup & generated folder structure
│
├── outputs/                    # Saved weights, checkpoints & plots land here
│   └── .gitkeep
│
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended; CPU training will be very slow)
- Kaggle account with API key

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/Anjishnu-Pal/Heritage_Healer.git
cd Heritage_Healer

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

All three approaches share the same CelebA dataset and the same paired `A/B` folder structure.

**Dataset:** [https://www.kaggle.com/datasets/jessicali9530/celeba-dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

See [data/README.md](data/README.md) for Kaggle API setup and the generated folder layout.

The `prepare_dataset.py` script in each model folder:
1. Installs the `kaggle` package and downloads CelebA
2. Damages each image (random scratches + Gaussian noise + grayscale conversion)
3. Saves `(damaged, clean)` pairs as:
   - `photo_restoration_dataset/train/A/` — damaged
   - `photo_restoration_dataset/train/B/` — clean original
   - `photo_restoration_dataset/val/A/` — damaged
   - `photo_restoration_dataset/val/B/` — clean original

> The VAE `train.py` includes dataset preparation inline — no separate script needed.

---

## Running the Scripts

### Diffusion Model (PyTorch)

```bash
# Step 1 — Build the dataset
python "src/Diffusion model/prepare_dataset.py"

# Step 2 — Train
python "src/Diffusion model/train.py"

# Step 3 — Restore an image
# Edit LAST_EPOCH and INPUT_IMAGE_PATH in the script first
python "src/Diffusion model/inference.py"
```

Saved weights: `diffusion_photo_restorer_epoch_N.pth`

---

### GAN / pix2pix (TensorFlow)

```bash
# Step 1 — Build the dataset
python src/GANs/prepare_dataset.py

# Step 2 — Train
python src/GANs/train.py

# Step 3 — Restore an image
# Edit INPUT_IMAGE_PATH and CHECKPOINT_DIR in the script first
python src/GANs/inference.py
```

Saved checkpoints: `training_checkpoints/ckpt-*`

---

### VAE (TensorFlow)

```bash
# Step 1+2 — Prepare data & train (combined)
python src/VAEs/train.py

# Step 3 — Restore an image
# Edit damaged_image_path in the script first
python src/VAEs/inference.py
```

Saved model: `model6.h5`

---

## Outputs

| Approach | Saved Artifact | Description |
|---|---|---|
| Diffusion | `diffusion_photo_restorer_epoch_N.pth` | UNet2D state dict after epoch N |
| GAN | `training_checkpoints/ckpt-*` | TF checkpoint (generator + discriminator) |
| VAE | `model6.h5` | Full Keras model (weights + architecture) |
| All | `restored_output.jpg` | Final restored image from inference |

---

## Key Hyperparameters

### Diffusion Model

| Parameter | Value | Notes |
|---|---|---|
| `IMG_SIZE` | 256 | Input/output resolution (px) |
| `NUM_IMAGES_TO_PROCESS` | 5000 | CelebA subset size |
| `TRAIN_SPLIT` | 0.9 | 90% train, 10% val |
| `train_batch_size` | 4 | Reduce to 2 if GPU OOM |
| `num_epochs` | 10 | Increase to 50–100 for best quality |
| `learning_rate` | 1e-4 | AdamW |
| `num_inference_steps` | 1000 | DDPM training timesteps |
| `num_validation_inference_steps` | 50 | Faster steps at validation |
| `in_channels` | 6 | 3 (noisy clean) + 3 (damaged condition) |

### GAN (pix2pix)

| Parameter | Value | Notes |
|---|---|---|
| `IMG_HEIGHT/WIDTH` | 256 | Input/output resolution (px) |
| `BATCH_SIZE` | 1 | Must be 1 for pix2pix |
| `EPOCHS` | 10 | Increase for better results |
| `LAMBDA` | 100 | L1 loss weight vs. adversarial loss |
| Generator LR | 2e-4 | Adam, β₁=0.5 |
| Discriminator LR | 2e-4 | Adam, β₁=0.5 |

### VAE

| Parameter | Value | Notes |
|---|---|---|
| `IMG_SIZE` | 256 | Input/output resolution (px) |
| `BATCH_SIZE` | 1 | Mini-batch size |
| `EPOCHS` | 10 | Training epochs |
| `LATENT_DIM` | 256 | Size of the latent space vector |
| `RECONSTRUCTION_WEIGHT` | 100 | L1 reconstruction loss weight vs. KL loss |
| Optimizer LR | 2e-4 | Adam, β₁=0.5 |

