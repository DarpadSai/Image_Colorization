import torch
import cv2
import os
from models.generator import UNetGenerator
from utils.data_loader import SAROpticalDataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
generator = UNetGenerator().to(device)
generator.load_state_dict(torch.load("outputs/checkpoints/gen_epoch_9.pth"))
generator.eval()

# Test dataset
test_dataset = SAROpticalDataset("data/test/sar", "data/test/optical")
test_img, _ = test_dataset[0]  # Take first test image
test_img = test_img.unsqueeze(0).to(device)

# Generate colorized image
with torch.no_grad():
    colorized_img = generator(test_img)

# Denormalize and save
colorized_img = (colorized_img.cpu().squeeze().numpy().transpose(1, 2, 0) + 1) * 127.5
colorized_img = colorized_img.astype(np.uint8)
cv2.imwrite("outputs/images/colorized_output.png", cv2.cvtColor(colorized_img, cv2.COLOR_RGB2BGR))
