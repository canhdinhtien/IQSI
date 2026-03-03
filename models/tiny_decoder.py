import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

        self.act = nn.SiLU()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + x

class TinyDecoder(nn.Module):
    def __init__(self, in_ch=4, base_ch=128):
        super().__init__()

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            ResBlock(base_ch),
            ResBlock(base_ch)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            ResBlock(base_ch),
            ResBlock(base_ch)
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=1),
            ResBlock(base_ch // 2),
            ResBlock(base_ch // 2)
        )

        self.norm_out = nn.GroupNorm(8, base_ch // 2)
        self.act_out = nn.SiLU()
        self.out_conv = nn.Conv2d(base_ch // 2, 3, 3, padding=1)

    def forward(self, z):
        h = self.in_conv(z)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.act_out(self.norm_out(h))
        img = self.out_conv(h)
        return torch.tanh(img)