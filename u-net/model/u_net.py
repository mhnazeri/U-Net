"""Baseline implementation of U-Net"""

import torch
import torch.nn as nn


def conv3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """forward path in each layer with padding"""
    layer = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
    )

    return layer


class UNet(nn.Module):
    """Basic U-Net"""

    def __init__(self, in_features: int = 3, out_features: int = 23):
        super().__init__()
        # Contracting path
        self.dwn_1 = conv3(in_features, 64)
        self.dwn_2 = conv3(64, 128)
        self.dwn_3 = conv3(128, 256)
        self.dwn_4 = conv3(256, 512)
        self.dwn_5 = conv3(512, 1024)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # expansive path
        self.up_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_forw_1 = conv3(1024, 512)
        self.up_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_forw_2 = conv3(512, 256)
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_forw_3 = conv3(256, 128)
        self.up_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_forw_4 = conv3(128, 64)

        # out layer
        self.out = nn.Conv2d(64, out_features, kernel_size=1)

    def forward(self, image):
        # Contracting path
        x_1 = self.dwn_1(image)
        x_2 = self.pool(x_1)

        x_2 = self.dwn_2(x_2)
        x_3 = self.pool(x_2)

        x_3 = self.dwn_3(x_3)
        x_4 = self.pool(x_3)

        x_4 = self.dwn_4(x_4)
        x_5 = self.pool(x_4)

        x_5 = self.dwn_5(x_5)

        # expansive path
        x = self.up_1(x_5, output_size=x_4.size())
        x = self.up_forw_1(torch.cat([x_4, x], 1))

        x = self.up_2(x, output_size=x_3.size())
        x = self.up_forw_2(torch.cat([x_3, x], 1))

        x = self.up_3(x, output_size=x_2.size())
        x = self.up_forw_3(torch.cat([x_2, x], 1))

        x = self.up_4(x, output_size=x_1.size())
        x = self.up_forw_4(torch.cat([x_1, x], 1))

        x = self.out(x)

        return x


class AutoEecoder(nn.Module):
    """Simple Autoencoder"""

    def __init__(self, in_features: int = 3, out_features: int = 23):
        super().__init__()
        # Contracting path
        self.dwn_1 = conv3(in_features, 64)
        self.dwn_2 = conv3(64, 128)
        self.dwn_3 = conv3(128, 256)
        self.dwn_4 = conv3(256, 512)
        self.dwn_5 = conv3(512, 1024)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # expansive path
        self.up_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_forw_1 = conv3(512, 512)
        self.up_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_forw_2 = conv3(256, 256)
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_forw_3 = conv3(128, 128)
        self.up_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_forw_4 = conv3(64, 64)

        # out layer
        self.out = nn.Conv2d(64, out_features, kernel_size=1)

    def forward(self, image):
        # Contracting path
        x = self.dwn_1(image)
        x_1_size = x.size()
        x = self.pool(x)

        x = self.dwn_2(x)
        x_2_size = x.size()
        x = self.pool(x)

        x = self.dwn_3(x)
        x_3_size = x.size()
        x = self.pool(x)

        x = self.dwn_4(x)
        x_4_size = x.size()
        x = self.pool(x)
        x = self.dwn_5(x)

        # expansive path
        x = self.up_1(x, output_size=x_4_size)
        x = self.up_forw_1(x)
        x = self.up_2(x, output_size=x_3_size)
        x = self.up_forw_2(x)
        x = self.up_3(x, output_size=x_2_size)
        x = self.up_forw_3(x)
        x = self.up_4(x, output_size=x_1_size)
        x = self.up_forw_4(x)
        x = self.out(x)

        return x


if __name__ == "__main__":
    pass
