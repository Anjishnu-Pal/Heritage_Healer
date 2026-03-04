import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps

# --- Reparameterization layer (must match train.py) ---
class Reparameterization(keras.layers.Layer):
    """The VAE sampling layer."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# --- VAE loss (must match train.py) ---
RECONSTRUCTION_WEIGHT = 100

def vae_loss(target, gen_output, z_mean, z_log_var):
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1))
    total_loss = (RECONSTRUCTION_WEIGHT * l1_loss) + kl_loss
    return total_loss

# --- Configuration ---
IMG_SIZE = 256
MODEL_PATH = 'model6.h5'
damaged_image_path = '/content/photo-restore-Lady_promotion_original.jpg' # <--- REPLACE WITH YOUR IMAGE PATH

# --- Load the saved model ---
try:
    loaded_vae_model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={'Reparameterization': Reparameterization, 'vae_loss': vae_loss}
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_vae_model = None

# --- Image preprocessing ---
def load_and_preprocess_image(image_path, img_size):
    """Loads, resizes, converts to grayscale, and normalizes an image for model prediction."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = ImageOps.fit(img, (img_size, img_size), Image.Resampling.LANCZOS)
        img_arr = np.array(img)

        # Convert to grayscale (1 channel) and then back to RGB (3 channels) for consistency
        # with the model's expected input shape, but with grayscale values replicated across channels.
        # Note: The model expects a 1-channel input, so we will only keep one channel after grayscale conversion.
        img_gray = ImageOps.grayscale(Image.fromarray(img_arr))
        img_input = np.array(img_gray) # This will be (IMG_SIZE, IMG_SIZE)

        # Add a channel dimension to make it (IMG_SIZE, IMG_SIZE, 1)
        img_input = np.expand_dims(img_input, axis=-1)

        # Normalize the image to [-1, 1]
        img_input = (img_input / 127.5) - 1

        # Add batch dimension
        img_input = np.expand_dims(img_input, axis=0)

        return img_input.astype(np.float32) # Ensure float32 type
    except Exception as e:
        print(f"Error loading and preprocessing {image_path}: {e}")
        return None

# --- Run Inference ---
if loaded_vae_model is not None and os.path.exists(damaged_image_path):
    # Load and preprocess the damaged image
    input_image_for_prediction = load_and_preprocess_image(damaged_image_path, IMG_SIZE)

    if input_image_for_prediction is not None:
        # Use the model to predict the restored image
        # The model outputs (reconstruction, z_mean, z_log_var), we only need the reconstruction
        restored_image_prediction, _, _ = loaded_vae_model.predict(input_image_for_prediction)

        # Denormalize the predicted image from [-1, 1] to [0, 1]
        restored_image_display = (restored_image_prediction[0] * 0.5 + 0.5)

        # Display the original damaged image and the restored image
        plt.figure(figsize=(10, 5))

        # For display, load the original image without model-specific preprocessing
        original_damaged_img = Image.open(damaged_image_path).convert('RGB')
        original_damaged_img = ImageOps.fit(original_damaged_img, (IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

        plt.subplot(1, 2, 1)
        plt.title('Original Damaged Image')
        plt.imshow(original_damaged_img)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Restored Image')
        plt.imshow(restored_image_display)
        plt.axis('off')

        plt.show()
    else:
        print("Could not preprocess the damaged image.")
else:
    if loaded_vae_model is None:
        print("VAE model not loaded.")
    if not os.path.exists(damaged_image_path):
        print(f"Damaged image not found at: {damaged_image_path}")
