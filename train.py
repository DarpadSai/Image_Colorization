import torch
import torch.nn as nn
from torch.optim import Adam
from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from utils.data_loader import get_dataloader
import os

# Hyperparameters
epochs = 50              # Train for longer; colorization tasks benefit from more training
lr = 0.0001               # Slightly lower learning rate for stable convergence
beta1 = 0.5               # Good for GANs; keeps momentum stable
lambda_l1 = 100           # L1 loss weight (default in Pix2Pix; helps preserve structure)
batch_size = 16           # Balanced to fit most GPUs and maintain training stability


# Paths
data_dir = "data/train"
output_dir = "outputs"
os.makedirs(output_dir + "/checkpoints", exist_ok=True)
os.makedirs(output_dir + "/images", exist_ok=True)

# Device
device = torch.device("cuda")

# Models
generator = UNetGenerator().to(device)
discriminator = PatchGANDiscriminator().to(device)

# Optimizers
g_optimizer = Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
d_optimizer = Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss functions
gan_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

# DataLoader
dataloader = get_dataloader(os.path.join(data_dir, "sar"), os.path.join(data_dir, "optical"), batch_size=batch_size)

# Training loop
for epoch in range(epochs):
    for i, (sar, optical) in enumerate(dataloader):
        sar, optical = sar.to(device), optical.to(device)

        # Train Discriminator
        d_optimizer.zero_grad()
        fake_optical = generator(sar)
        real_output = discriminator(sar, optical)
        fake_output = discriminator(sar, fake_optical.detach())
        
        d_loss_real = gan_loss(real_output, torch.ones_like(real_output))
        d_loss_fake = gan_loss(fake_output, torch.zeros_like(fake_output))
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        fake_output = discriminator(sar, fake_optical)
        g_gan_loss = gan_loss(fake_output, torch.ones_like(fake_output))
        g_l1_loss = l1_loss(fake_optical, optical) * lambda_l1
        g_loss = g_gan_loss + g_l1_loss
        g_loss.backward()
        g_optimizer.step()

        if i % 16 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                  f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

    # Save checkpoint
    torch.save(generator.state_dict(), f"{output_dir}/checkpoints/gen_epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"{output_dir}/checkpoints/disc_epoch_{epoch}.pth")
