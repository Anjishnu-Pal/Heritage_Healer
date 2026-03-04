import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import subprocess # Use subprocess to call the Kaggle API
import random
from tqdm import tqdm

# --- Configuration ---
# Kaggle dataset identifier
KAGGLE_DATASET_ID = "jessicali9530/celeba-dataset"
# The folder containing images after unzipping
# The Kaggle dataset unzips into 'img_align_celeba/img_align_celeba/'
SOURCE_IMG_FOLDER = "img_align_celeba/img_align_celeba"
OUTPUT_DIR = "photo_restoration_dataset"
IMG_SIZE = 256
TRAIN_SPLIT = 0.9
# Let's use a smaller subset for a quick start, e.g., 5000 images
NUM_IMAGES_TO_PROCESS = 5000

# --- Helper Functions for "Damaging" Images (These are unchanged) ---

def add_noise(img):
    """Adds Gaussian noise to an image."""
    mean = 0
    var = random.uniform(5, 15)
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, img.shape)
    noisy_img = np.clip(img + gaussian, 0, 255).astype(np.uint8)
    return noisy_img

def add_scratches(img):
    """Overlays random lines as simple scratches."""
    scratch_img = img.copy()
    height, width, _ = scratch_img.shape
    num_scratches = random.randint(5, 15)
    for _ in range(num_scratches):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        color_val = random.randint(0, 50)
        color = (color_val, color_val, color_val)
        thickness = random.randint(1, 2)
        cv2.line(scratch_img, (x1, y1), (x2, y2), color, thickness)
    return scratch_img

def process_and_damage_image(img_path):
    """Loads, resizes, and applies a series of damages to an image."""
    try:
        img = Image.open(img_path).convert('RGB')
        img = ImageOps.fit(img, (IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        clean_img_arr = np.array(img)

        # Create Damaged Version
        damaged_img = add_scratches(clean_img_arr)
        damaged_img = add_noise(damaged_img)

        damaged_img_pil = Image.fromarray(damaged_img)
        damaged_img_pil = ImageOps.grayscale(damaged_img_pil).convert('RGB')

        return np.array(damaged_img_pil), clean_img_arr
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None

# --- Main Script ---

# 1. Install Kaggle and Download the Dataset using the API
print("Checking for Kaggle API...")
try:
    subprocess.run(['pip', 'install', 'kaggle'], check=True)
    print("Downloading and unzipping CelebA dataset from Kaggle...")
    # Command to download and unzip the dataset to the current directory
    subprocess.run([
        'kaggle', 'datasets', 'download', '-d', KAGGLE_DATASET_ID, '-p', '.', '--unzip'
    ], check=True)
    print("Dataset downloaded successfully.")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("="*50)
    print("ERROR: Kaggle API setup is required.")
    print("Please follow the one-time setup instructions to get your `kaggle.json` API token.")
    print("="*50)
    exit() # Exit if Kaggle setup is not done

# 2. Setup Output Directories
os.makedirs(os.path.join(OUTPUT_DIR, "train/A"), exist_ok=True) # Damaged
os.makedirs(os.path.join(OUTPUT_DIR, "train/B"), exist_ok=True) # Clean
os.makedirs(os.path.join(OUTPUT_DIR, "val/A"), exist_ok=True)   # Damaged
os.makedirs(os.path.join(OUTPUT_DIR, "val/B"), exist_ok=True)   # Clean

# 3. Process Images
if not os.path.exists(SOURCE_IMG_FOLDER):
    print(f"ERROR: Source image folder not found at '{SOURCE_IMG_FOLDER}'")
    print("Please check the unzipped folder structure from Kaggle.")
    exit()

image_files = [f for f in os.listdir(SOURCE_IMG_FOLDER) if f.endswith(('.jpg', '.png'))]
image_files = image_files[:NUM_IMAGES_TO_PROCESS]
random.shuffle(image_files)

split_index = int(len(image_files) * TRAIN_SPLIT)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

print(f"Processing {len(train_files)} training images and {len(val_files)} validation images...")

def save_images(file_list, split_name):
    for filename in tqdm(file_list, desc=f"Creating {split_name} set"):
        img_path = os.path.join(SOURCE_IMG_FOLDER, filename)
        damaged_img, clean_img = process_and_damage_image(img_path)

        if damaged_img is not None and clean_img is not None:
            Image.fromarray(damaged_img).save(os.path.join(OUTPUT_DIR, f"{split_name}/A", filename))
            Image.fromarray(clean_img).save(os.path.join(OUTPUT_DIR, f"{split_name}/B", filename))

save_images(train_files, "train")
save_images(val_files, "val")

print("="*50)
print("Dataset preparation complete!")
print(f"Your training and validation sets are ready in the '{OUTPUT_DIR}' folder.")
print("="*50)
