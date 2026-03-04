import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Configuration ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
CHECKPOINT_DIR = './training_checkpoints' # Path to your saved checkpoints
INPUT_IMAGE_PATH = '/photo-restore-Lady_promotion_original.jpg' # CHANGE THIS

# --- Re-create the Generator Model ---
# Note: You must define the same model architecture as in training.
# It's best practice to have this in a separate models.py file.
# For simplicity, we redefine it here.
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

def Generator():
    inputs = layers.Input(shape=[256, 256, 3])
    # ... (Copy the exact same U-Net architecture from train.py here) ...
    down_stack = [ downsample(64, 4, False), downsample(128, 4), downsample(256, 4), downsample(512, 4), downsample(512, 4), downsample(512, 4), downsample(512, 4), downsample(512, 4),]
    up_stack = [ upsample(512, 4, True), upsample(512, 4, True), upsample(512, 4, True), upsample(512, 4), upsample(256, 4), upsample(128, 4), upsample(64, 4),]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(3, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    x = last(x)
    return keras.Model(inputs=inputs, outputs=x)


# --- Load Model and Run Prediction ---
# 1. Create a generator instance
generator = Generator()

# 2. Load the latest checkpoint
checkpoint = tf.train.Checkpoint(generator=generator)
checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR)).expect_partial()

# 3. Load and preprocess the input image
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    # Normalize to [-1, 1]
    image = (image / 127.5) - 1
    # Expand dims to create a batch of 1
    image = tf.expand_dims(image, axis=0)
    return image

input_image = load_and_preprocess_image(INPUT_IMAGE_PATH)

# 4. Generate the restored image
prediction = generator(input_image, training=True)

# 5. Post-process and display the result
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(input_image[0] * 0.5 + 0.5)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Restored Image")
# De-normalize from [-1, 1] to [0, 1]
plt.imshow(prediction[0] * 0.5 + 0.5)
plt.axis('off')

# Save the result
restored_image_array = (prediction[0].numpy() * 0.5 + 0.5) * 255
restored_image_array = restored_image_array.astype(np.uint8)
Image.fromarray(restored_image_array).save("restored_output.jpg")

plt.show()
