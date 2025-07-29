import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        
        def down_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        def up_block(in_channels, out_channels, dropout=False):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)]
            layers.append(nn.BatchNorm2d(out_channels))
            if dropout:
                layers.append(nn.Dropout(0.5))
            layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        # Encoder
        self.down1 = down_block(1, 64, normalize=False)  # Input: SAR grayscale
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)

        # Decoder
        self.up1 = up_block(512, 256, dropout=True)
        self.up2 = up_block(512, 128, dropout=True)
        self.up3 = up_block(256, 64)
        self.up4 = nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        u4 = self.up4(torch.cat([u3, d1], dim=1))
        return self.tanh(u4)
