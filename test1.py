import torch
import cv2
import os
import matplotlib.pyplot as plt
from models.generator import UNetGenerator
from utils.data_loader import SAROpticalDataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
generator = UNetGenerator().to(device)
generator.load_state_dict(torch.load("outputs/checkpoints/gen_epoch_49.pth", map_location=device, weights_only=True))
generator.eval()

# Test dataset
test_dataset = SAROpticalDataset("data/test/sar", "data/test/optical")
print(f"Dataset length: {len(test_dataset)}")
if len(test_dataset) == 0:
    raise ValueError("Error: Test dataset is empty or paths are incorrect.")

test_img, _ = test_dataset[30]  # Take first test image
test_img = test_img.unsqueeze(0).to(device)

# Generate colorized image
with torch.no_grad():
    colorized_img = generator(test_img)
    print("Colorized image shape:", colorized_img.shape)

# Denormalize and save
colorized_img = np.clip((colorized_img.cpu().squeeze().numpy().transpose(1, 2, 0) + 1) * 127.5, 0, 255)
colorized_img = colorized_img.astype(np.uint8)

# Ensure output directory exists
os.makedirs("outputs/images", exist_ok=True)

cv2.imwrite("outputs/images/colorized_output.png", cv2.cvtColor(colorized_img, cv2.COLOR_RGB2BGR))
print("Image saved successfully.")

# Display the image
image_path = "outputs/images/colorized_output.png"
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found at", image_path)
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')  # Remove axis for better visualization
    plt.show()
