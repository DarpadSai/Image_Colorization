import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class SAROpticalDataset(Dataset):
    def __init__(self, sar_dir, optical_dir, transform=None):
        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.transform = transform
        self.sar_images = sorted(os.listdir(sar_dir))
        self.optical_images = sorted(os.listdir(optical_dir))

    def __len__(self):
        return len(self.sar_images)

    def __getitem__(self, idx):
        sar_path = os.path.join(self.sar_dir, self.sar_images[idx])
        optical_path = os.path.join(self.optical_dir, self.optical_images[idx])

        sar_img = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
        optical_img = cv2.imread(optical_path, cv2.IMREAD_COLOR)
        sar_img = cv2.resize(sar_img, (256, 256))
        optical_img = cv2.resize(optical_img, (256, 256))

        # Normalize to [-1, 1]
        sar_img = (sar_img / 127.5) - 1.0
        optical_img = (optical_img / 127.5) - 1.0

        # Convert to tensor
        sar_img = torch.tensor(sar_img, dtype=torch.float32).unsqueeze(0)  # [1, 256, 256]
        optical_img = torch.tensor(optical_img, dtype=torch.float32).permute(2, 0, 1)  # [3, 256, 256]

        return sar_img, optical_img

def get_dataloader(sar_dir, optical_dir, batch_size=4):
    dataset = SAROpticalDataset(sar_dir, optical_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
