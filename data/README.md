# Data Directory

All three model approaches (Diffusion, GAN, VAE) share the same CelebA dataset and the same paired `A/B` folder structure.

The dataset is downloaded automatically via the Kaggle API by running `prepare_dataset.py` inside any model folder (or `src/VAEs/train.py` which includes data prep inline).

## Dataset Source

**CelebA (Large-scale CelebFaces Attributes Dataset)**
> [https://www.kaggle.com/datasets/jessicali9530/celeba-dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

## Kaggle API Setup

1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings) → *API* → *Create New Token*
2. Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`
3. Restrict its permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

## How Damage Is Simulated

Each clean CelebA image is synthetically aged in three steps:
1. **Random scratches** — `cv2.line` draws 5–15 dark lines at random angles
2. **Gaussian noise** — random variance (σ² ∈ [5, 15]) added to all pixels
3. **Grayscale conversion** — image converted to grayscale and back to RGB (simulates fading/discoloration)

The resulting `(damaged, clean)` pair is saved to folders `A/` and `B/` respectively.

## Generated Structure

After running the dataset script, the following folder is created in the project root:

```
photo_restoration_dataset/
├── train/
│   ├── A/          ← Damaged images (grayscale + scratches + noise)  [4500 images]
│   │   ├── 000001.jpg
│   │   └── ...
│   └── B/          ← Clean original color images                     [4500 images]
│       ├── 000001.jpg
│       └── ...
└── val/
    ├── A/          ← Damaged images                                  [500 images]
    │   └── ...
    └── B/          ← Clean original color images                     [500 images]
        └── ...
```

- **Diffusion model** reads both `A/` (damaged condition) and `B/` (clean target) as 3-channel RGB.
- **GAN** reads `B/` for file listing and dynamically loads the matching `A/` image; both are 3-channel RGB.
- **VAE** reads `B/` for file listing and loads the matching `A/` image as 1-channel grayscale (model input) with `B/` as 3-channel RGB (target).

> **Note:** The raw CelebA download (`img_align_celeba/`) and the generated dataset (`photo_restoration_dataset/`) are excluded from version control via `.gitignore` due to their size.
