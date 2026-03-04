import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import subprocess
import random
from tqdm import tqdm
import time

# --- Configuration ---
# Data preparation settings
KAGGLE_DATASET_ID = "jessicali9530/celeba-dataset"
SOURCE_IMG_FOLDER = "img_align_celeba/img_align_celeba"
OUTPUT_DIR = "photo_restoration_dataset"
IMG_SIZE = 256
TRAIN_SPLIT = 0.9
NUM_IMAGES_TO_PROCESS = 5000  # Use a subset for a quicker run

# Model and training settings
BUFFER_SIZE = 400
BATCH_SIZE = 1
EPOCHS = 10  # Increased epochs for better results
LATENT_DIM = 256
RECONSTRUCTION_WEIGHT = 100

def add_noise(img):
    """Adds Gaussian noise to a NumPy image array."""
    mean = 0
    var = random.uniform(5, 15)
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, img.shape)
    noisy_img = np.clip(img + gaussian, 0, 255).astype(np.uint8)
    return noisy_img

def add_scratches(img):
    """Overlays random lines as simple scratches on a NumPy image array."""
    scratch_img = img.copy()
    h, w, _ = scratch_img.shape
    num_scratches = random.randint(5, 15)
    for _ in range(num_scratches):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        color_val = random.randint(0, 50)
        color = (color_val, color_val, color_val)
        thickness = random.randint(1, 2)
        cv2.line(scratch_img, (x1, y1), (x2, y2), color, thickness)
    return scratch_img

def process_and_damage_image(img_path):
    """Loads, resizes, and applies damages to an image."""
    try:
        img = Image.open(img_path).convert('RGB')
        img = ImageOps.fit(img, (IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        clean_img_arr = np.array(img)

        # Create Damaged Version
        damaged_img = add_scratches(clean_img_arr)
        damaged_img = add_noise(damaged_img)

        # Convert to faded/grayscale
        damaged_img_pil = Image.fromarray(damaged_img)
        damaged_img_pil = ImageOps.grayscale(damaged_img_pil).convert('RGB')

        return np.array(damaged_img_pil), clean_img_arr
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None

def prepare_dataset():
    """Downloads, processes, and saves the image dataset."""
    print("STEP 1: Preparing the Dataset")
    # 1. Download the Dataset using the Kaggle API
    try:
        print(">>> Installing Kaggle API...")
        subprocess.run(['pip', 'install', 'kaggle'], check=True, capture_output=True)
        print(">>> Downloading and unzipping CelebA dataset...")
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', KAGGLE_DATASET_ID, '-p', '.', '--unzip'],
            check=True, capture_output=True
        )
        print(">>> Dataset downloaded successfully.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("="*50)
        print("ERROR: Kaggle API setup is required.")
        print("Please upload your `kaggle.json` API token to the environment.")
        print("="*50)
        return False

    # 2. Setup Output Directories
    os.makedirs(os.path.join(OUTPUT_DIR, "train/A"), exist_ok=True) # Damaged
    os.makedirs(os.path.join(OUTPUT_DIR, "train/B"), exist_ok=True) # Clean
    os.makedirs(os.path.join(OUTPUT_DIR, "val/A"), exist_ok=True)   # Damaged
    os.makedirs(os.path.join(OUTPUT_DIR, "val/B"), exist_ok=True)   # Clean

    # 3. Process Images
    if not os.path.exists(SOURCE_IMG_FOLDER):
        print(f"ERROR: Source folder not found at '{SOURCE_IMG_FOLDER}'")
        return False

    image_files = [f for f in os.listdir(SOURCE_IMG_FOLDER) if f.endswith(('.jpg', '.png'))]
    image_files = image_files[:NUM_IMAGES_TO_PROCESS]
    random.shuffle(image_files)

    split_index = int(len(image_files) * TRAIN_SPLIT)
    train_files, val_files = image_files[:split_index], image_files[split_index:]

    print(f">>> Processing {len(train_files)} training and {len(val_files)} validation images...")

    def save_images(file_list, split_name):
        for filename in tqdm(file_list, desc=f"Creating {split_name} set"):
            img_path = os.path.join(SOURCE_IMG_FOLDER, filename)
            damaged_img, clean_img = process_and_damage_image(img_path)
            if damaged_img is not None and clean_img is not None:
                Image.fromarray(damaged_img).save(os.path.join(OUTPUT_DIR, f"{split_name}/A", filename))
                Image.fromarray(clean_img).save(os.path.join(OUTPUT_DIR, f"{split_name}/B", filename))

    save_images(train_files, "train")
    save_images(val_files, "val")
    print(">>> Dataset preparation complete!")
    return True

# ==============================================================================
# PART 2: TENSORFLOW DATA PIPELINE & VAE MODEL
# ==============================================================================
print("\nSTEP 2: Defining the VAE Model and Data Pipeline")

# --- TensorFlow Data Loading ---
def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image, channels=3) # Target is always 3-channel

    input_path = tf.strings.regex_replace(image_file, "/B/", "/A/")
    input_image = tf.io.read_file(input_path)
    input_image = tf.io.decode_jpeg(input_image, channels=1) # Input is 1-channel

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(image, tf.float32)
    return input_image, real_image

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image

def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_SIZE, IMG_SIZE)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image

# --- VAE U-Net Model ---
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm: result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(layers.BatchNormalization())
    if apply_dropout: result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

class Reparameterization(layers.Layer):
    """The VAE sampling layer."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def VAE():
    inputs = layers.Input(shape=[IMG_SIZE, IMG_SIZE, 1]) # 1-channel input

    # Encoder
    down1 = downsample(64, 4, apply_batchnorm=False)(inputs) # (None, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (None, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (None, 32, 32, 256)
    down4 = downsample(512, 4)(down3) # (None, 16, 16, 512)
    down5 = downsample(512, 4)(down4) # (None, 8, 8, 512)
    down6 = downsample(512, 4)(down5) # (None, 4, 4, 512)
    down7 = downsample(512, 4)(down6) # (None, 2, 2, 512) - Bottleneck

    # --- Bottleneck and VAE Core ---
    shape_before_flatten = tf.keras.backend.int_shape(down7)[1:]
    flatten = layers.Flatten()(down7)
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(flatten)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(flatten)
    z = Reparameterization()([z_mean, z_log_var])

    shape_product = np.prod(shape_before_flatten)
    x = layers.Dense(shape_product)(z)
    x = layers.Reshape(shape_before_flatten)(x)

    # Decoder
    up1 = upsample(512, 4, apply_dropout=True)(x)    # (None, 4, 4, 512)
    concat1 = layers.Concatenate()([up1, down6])      # Concatenate with down6 (4x4)

    up2 = upsample(512, 4, apply_dropout=True)(concat1) # (None, 8, 8, 512)
    concat2 = layers.Concatenate()([up2, down5])         # Concatenate with down5 (8x8)

    up3 = upsample(512, 4)(concat2)                   # (None, 16, 16, 512)
    concat3 = layers.Concatenate()([up3, down4])       # Concatenate with down4 (16x16)

    up4 = upsample(256, 4)(concat3)                   # (None, 32, 32, 256)
    concat4 = layers.Concatenate()([up4, down3])       # Concatenate with down3 (32x32)

    up5 = upsample(128, 4)(concat4)                   # (None, 64, 64, 128)
    concat5 = layers.Concatenate()([up5, down2])       # Concatenate with down2 (64x64)

    up6 = upsample(64, 4)(concat5)                    # (None, 128, 128, 64)
    concat6 = layers.Concatenate()([up6, down1])       # Concatenate with down1 (128x128)

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(
        3, 4, strides=2, padding='same', # 3-channel output
        kernel_initializer=initializer, activation='tanh'
    )
    reconstruction = last(concat6) # (None, 256, 256, 3)

    return keras.Model(inputs=inputs, outputs=[reconstruction, z_mean, z_log_var])

# --- VAE Loss Function ---
def vae_loss(target, gen_output, z_mean, z_log_var):
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1))
    total_loss = (RECONSTRUCTION_WEIGHT * l1_loss) + kl_loss
    return total_loss

# --- Training Step ---
@tf.function
def train_step(model, input_image, target, optimizer):
    with tf.GradientTape() as tape:
        gen_output, z_mean, z_log_var = model(input_image, training=True)
        loss = vae_loss(target, gen_output, z_mean, z_log_var)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# --- Image Generation for Visualization ---
def generate_images(model, test_input, tar):
    prediction, _, _ = model(test_input, training=False)
    plt.figure(figsize=(15, 5))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input (Damaged)', 'Ground Truth (Clean)', 'Predicted (Restored)']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        img = display_list[i] * 0.5 + 0.5 # Denormalize from [-1, 1] to [0, 1]
        if i == 0:
            plt.imshow(img, cmap='gray') # Show input in grayscale
        else:
            plt.imshow(img)
        plt.axis('off')
    plt.show()

def main():
    if not prepare_dataset():
        return None # Return None if dataset preparation fails

    print("\nSTEP 3: Starting Model Training")

    # --- Initialize Model and Optimizer ---
    vae_model = VAE()
    optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # --- Load Datasets ---
    train_dataset = tf.data.Dataset.list_files(os.path.join(OUTPUT_DIR, "train/B/*.jpg"))
    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    val_dataset = tf.data.Dataset.list_files(os.path.join(OUTPUT_DIR, "val/B/*.jpg"))
    val_dataset = val_dataset.map(load_image_train).batch(BATCH_SIZE)

    # --- Main Training Loop ---
    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0

        print(f"Epoch {epoch + 1}/{EPOCHS}")

        for n, (input_image, target) in tqdm(enumerate(train_dataset), total=tf.data.experimental.cardinality(train_dataset).numpy(), desc="Training"):
            loss = train_step(vae_model, input_image, target, optimizer)
            total_loss += loss

        avg_loss = total_loss / (n + 1)
        print(f"  - Time: {time.time()-start:.2f}s, Average Loss: {avg_loss:.4f}")

        # Visualize progress on one validation image
        for example_input, example_target in val_dataset.take(1):
            generate_images(vae_model, example_input, example_target)

    print("\nTraining Complete!")
    return vae_model # Return the trained model

if __name__ == '__main__':
    # Call main and store the returned model in a variable
    trained_vae_model = main()

    # Save the entire model
    if trained_vae_model is not None:
        trained_vae_model.save('model6.h5')
        print("Model saved successfully as model6.h5")
    else:
        print("Model not trained or found. Cannot save.")
