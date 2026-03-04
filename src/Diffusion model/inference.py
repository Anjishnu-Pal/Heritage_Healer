import torch
from diffusers import UNet2DModel, DDPMScheduler
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm

# --- 1. Configuration (MUST MATCH TRAINING) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256 # From Cell 1
MODEL_SAVE_PATH = "diffusion_photo_restorer" # From Cell 2
LAST_EPOCH = 10 # !! CHANGE THIS to the epoch number you saved (e.g., 30)

# >> TUNE THIS <<: Path to the weights file
MODEL_WEIGHTS_PATH = f"{MODEL_SAVE_PATH}_epoch_{LAST_EPOCH}.pth"

# >> TUNE THIS <<: Path to your "old" input image
# We'll grab the first image from the validation 'A' (damaged) folder as an example
INPUT_IMAGE_PATH = os.path.join("photo_restoration_dataset", "val/A", os.listdir("photo_restoration_dataset/val/A")[0])
# You can also upload your own image and set the path here, e.g., "/content/my_old_photo.jpg"

print(f"Using device: {device}")
print(f"Loading weights from: {MODEL_WEIGHTS_PATH}")
print(f"Restoring image: {INPUT_IMAGE_PATH}")

# --- 2. Re-load Model Architecture (MUST MATCH TRAINING) ---
print("Re-building model architecture...")
model = UNet2DModel(
    sample_size=IMG_SIZE,
    in_channels=6,  # 3 (noisy_clean) + 3 (damaged_condition)
    out_channels=3, # Predict the 3-channel noise
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D",
        "AttnDownBlock2D", "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D",
        "UpBlock2D", "UpBlock2D"
    ),
)

# --- 3. Load Trained Weights ---
if not os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"ERROR: Model weights not found at '{MODEL_WEIGHTS_PATH}'")
    print("Please check the 'LAST_EPOCH' variable and the 'MODEL_SAVE_PATH' variable.")
else:
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode (important!)
    print("Model weights loaded successfully.")

    # --- 4. Load and Process the "Old" Image ---

    # Preprocessing (MUST MATCH TRAINING)
    preprocess = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Normalize to [-1, 1]
        ]
    )

    # Load the damaged (condition) image
    damaged_image_pil = Image.open(INPUT_IMAGE_PATH).convert("RGB")
    damaged_image_tensor = preprocess(damaged_image_pil).to(device)

    # Add a batch dimension (B, C, H, W)
    # Our model expects a batch, even for a single image
    condition = damaged_image_tensor.unsqueeze(0) # Shape: [1, 3, 256, 256]

    # --- 5. Run the Inference (Denoising) Pipeline ---
    print("Starting denoising process...")

    # Set up the scheduler
    num_inference_steps = 50 # Number of steps to denoise (faster than training)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    noise_scheduler.set_timesteps(num_inference_steps)

    # Start with pure random noise (this will become our restored image)
    restored_image = torch.randn_like(condition) # Shape: [1, 3, 256, 256]

    # Denoising loop
    for t in tqdm(noise_scheduler.timesteps):
        with torch.no_grad(): # No need to track gradients

            # 1. Create the 6-channel input
            # (current_noisy_output, damaged_condition)
            model_input = torch.cat([restored_image, condition], dim=1)

            # 2. Predict the noise
            noise_pred = model(model_input, t).sample

            # 3. Use scheduler to denoise one step
            restored_image = noise_scheduler.step(noise_pred, t, restored_image).prev_sample

    print("Denoising complete.")

    # --- 6. Post-process and Display Results ---

    def tensor_to_pil(tensor_img):
        """Converts a [-1, 1] tensor to a PIL Image."""
        # De-normalize from [-1, 1] to [0, 1]
        tensor_img = (tensor_img + 1) / 2
        # Clamp values to [0, 1] just in case
        tensor_img = tensor_img.clamp(0, 1)
        # Remove batch dim, move to CPU, and convert
        pil_img = transforms.ToPILImage()(tensor_img.squeeze(0).cpu())
        return pil_img

    # Convert our tensors back to PIL Images
    restored_pil = tensor_to_pil(restored_image)

    # We can just use the 'damaged_image_pil' we loaded earlier

    # Display side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(damaged_image_pil)
    axs[0].set_title("Old Damaged Image (Input)")
    axs[0].axis("off")

    axs[1].imshow(restored_pil)
    axs[1].set_title("Restored Image (Output)")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

    # Save the restored image
    restored_pil.save("restored_output.jpg")
    print("Restored image saved as 'restored_output.jpg'")
