import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()
        
        def disc_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            disc_block(4, 64, normalize=False),  # Input: SAR (1) + Optical (3)
            disc_block(64, 128),
            disc_block(128, 256),
            nn.Conv2d(256, 1, 4, padding=1)  # Output: Patch-wise prediction
        )

    def forward(self, sar, optical):
        x = torch.cat([sar, optical], dim=1)  # Concatenate SAR and optical
        return self.model(x)
