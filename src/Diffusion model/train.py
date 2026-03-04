print("🚀 Starting Cell 2: Model Training...")

# --- Part 1: Install Libraries & Imports ---
import subprocess
print("Installing libraries (diffusers, accelerate, transformers)...")
subprocess.run(['pip', 'install', 'diffusers[torch]', 'accelerate', 'transformers', '-q'], check=True)

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

# --- Part 2: Configuration ---
config = {
    "dataset_path": "photo_restoration_dataset",
    "image_size": 256,  # Must match IMG_SIZE from Cell 1
    "train_batch_size": 4, # Reduce if you get OOM errors
    "eval_batch_size": 4,
    "num_epochs": 10,     # Increase for better results (e.g., 50-100)
    "learning_rate": 1e-4,
    "model_save_path": "diffusion_photo_restorer",
    "num_inference_steps": 1000, # Number of steps for training scheduler
    "num_validation_inference_steps": 50, # Steps for validation (faster)
}

# --- Part 3: Create the PyTorch Dataset ---
class PairedImageDataset(Dataset):
    """
    Loads paired images from folders A (Damaged/Condition) and B (Clean/Target).
    """
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.dir_A = os.path.join(root_dir, split, "A") # Damaged
        self.dir_B = os.path.join(root_dir, split, "B") # Clean

        self.image_files = [f for f in os.listdir(self.dir_A) if os.path.exists(os.path.join(self.dir_B, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_A_path = os.path.join(self.dir_A, img_name)
        img_B_path = os.path.join(self.dir_B, img_name)

        img_A = Image.open(img_A_path).convert("RGB") # Damaged
        img_B = Image.open(img_B_path).convert("RGB") # Clean

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return img_A, img_B

# Define standard diffusion model preprocessing
# We normalize to [-1, 1]
preprocess = transforms.Compose(
    [
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# --- Part 4: Setup Dataloaders, Model, Scheduler ---
print("Setting up Dataloaders, Model, and Scheduler...")

train_dataset = PairedImageDataset(config["dataset_path"], split="train", transform=preprocess)
train_dataloader = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True)

# Use validation set for testing
val_dataset = PairedImageDataset(config["dataset_path"], split="val", transform=preprocess)
val_dataloader = DataLoader(val_dataset, batch_size=config["eval_batch_size"], shuffle=False)

# Model: UNet2DModel
# We use 6 input channels: 3 for the noisy clean image (xt) + 3 for the damaged condition image (c)
model = UNet2DModel(
    sample_size=config["image_size"],
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

# Scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=config["num_inference_steps"])

# Optimizer and LR Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=(len(train_dataloader) * config["num_epochs"]),
)

# --- Part 5: Accelerator for Training ---
accelerator = Accelerator(
    mixed_precision="fp16",  # Use mixed precision for faster training
    log_with="tensorboard",
    project_dir=os.path.join(config["model_save_path"], "logs")
)

model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

print(f"Starting training for {config['num_epochs']} epochs...")
# --- Part 6: Training Loop ---

global_step = 0
for epoch in range(config["num_epochs"]):
    model.train()
    progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config['num_epochs']}")

    for step, (damaged_images, clean_images) in enumerate(train_dataloader):
        # clean_images = B (target), damaged_images = A (condition)

        # 1. Sample noise
        noise = torch.randn_like(clean_images)

        # 2. Sample timesteps
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],),
            device=accelerator.device
        ).long()

        # 3. Add noise to the *clean* images (forward process)
        noisy_clean_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # 4. Concatenate noisy clean images and damaged condition images
        # This is our 6-channel model input
        model_input = torch.cat([noisy_clean_images, damaged_images], dim=1)

        # 5. Predict the noise
        with accelerator.accumulate(model):
            predicted_noise = model(model_input, timesteps).sample

            # 6. Calculate loss (MSE between predicted noise and actual noise)
            loss = F.mse_loss(predicted_noise, noise)

            # 7. Backpropagate
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item())
        global_step += 1

    progress_bar.close()
    accelerator.wait_for_everyone()

    # --- Part 7: Validation & Saving (End of Epoch) ---
    if accelerator.is_main_process:
        print(f"Epoch {epoch+1} complete. Running validation...")

        # Get a batch of validation data
        val_batch = next(iter(val_dataloader))
        val_damaged, val_clean = val_batch
        val_damaged = val_damaged.to(accelerator.device)

        # Start with random noise for the *output*
        generated_images = torch.randn_like(val_clean).to(accelerator.device)

        # Set scheduler for validation
        noise_scheduler.set_timesteps(config["num_validation_inference_steps"])

        # Manual sampling loop
        for t in tqdm(noise_scheduler.timesteps, desc="Generating validation images"):
            with torch.no_grad():
                model.eval()
                # Create the 6-channel input: (current_noisy_output, damaged_condition)
                model_input = torch.cat([generated_images, val_damaged], dim=1)

                # Predict noise
                noise_pred = model(model_input, t).sample

                # Use scheduler to denoise one step
                generated_images = noise_scheduler.step(noise_pred, t, generated_images).prev_sample

        # --- Visualization ---
        def tensor_to_pil(tensor_img):
            # Denormalize from [-1, 1] to [0, 1]
            tensor_img = (tensor_img + 1) / 2
            # Clamp, move to CPU, convert to 0-255 uint8
            pil_img = transforms.ToPILImage()(tensor_img).convert("RGB")
            return pil_img

        # Show N images: Damaged | Generated | Clean
        N_IMAGES_TO_SHOW = 3
        fig, axs = plt.subplots(N_IMAGES_TO_SHOW, 3, figsize=(15, N_IMAGES_TO_SHOW * 5))

        for i in range(N_IMAGES_TO_SHOW):
            if i >= len(val_damaged): break # In case batch is smaller

            dam_pil = tensor_to_pil(val_damaged[i].cpu())
            gen_pil = tensor_to_pil(generated_images[i].cpu())
            cln_pil = tensor_to_pil(val_clean[i].cpu()) # val_clean was not moved to device

            axs[i, 0].imshow(dam_pil)
            axs[i, 0].set_title(f"Damaged (Input) {i}")
            axs[i, 0].axis("off")

            axs[i, 1].imshow(gen_pil)
            axs[i, 1].set_title(f"Restored (Generated) {i}")
            axs[i, 1].axis("off")

            axs[i, 2].imshow(cln_pil)
            axs[i, 2].set_title(f"Original (Ground Truth) {i}")
            axs[i, 2].axis("off")

        plt.tight_layout()
        plt.show()

        # Save the final model
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), f"{config['model_save_path']}_epoch_{epoch+1}.pth")
        print(f"Model saved as {config['model_save_path']}_epoch_{epoch+1}.pth")

print("="*50)
print("✅ Cell 2 Complete: Model training finished!")
print(f"Final model weights are saved with the prefix '{config['model_save_path']}'")
print("="*50)
