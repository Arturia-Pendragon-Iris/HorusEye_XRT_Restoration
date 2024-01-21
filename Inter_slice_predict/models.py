import torch
import torch.nn as nn
import numpy as np
import pydicom
import torch.nn.functional as F


class Residual_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernal=3, padding=1):
        super(Residual_Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                      kernel_size=(kernal, kernal), stride=(1, 1), padding=padding, bias=False),
            nn.InstanceNorm2d(self.out_channel, affine=True),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel,
                      kernel_size=(kernal, kernal), stride=(1, 1), padding=padding, bias=False),
            nn.InstanceNorm2d(self.out_channel, affine=True),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        output = x + self.conv2(x)
        return output


class UNet_2D(nn.Module):
    def __init__(self, inchannels=2, outchannels=1):
        super(UNet_2D, self).__init__()

        self.in_channels = inchannels
        self.out_channels = outchannels

        scale = 64

        self.layer_0 = Residual_Block(self.in_channels, scale)
        self.layer_1 = Residual_Block(scale, scale * 2)
        self.layer_2 = Residual_Block(scale * 2, scale * 4)
        self.layer_3 = Residual_Block(scale * 4, scale * 8)
        self.layer_4 = Residual_Block(scale * 8, scale * 4)
        self.layer_5 = Residual_Block(scale * 4, scale * 2)
        self.layer_6 = Residual_Block(scale * 2, scale)
        self.upscale = Residual_Block(scale, 32)

        self.upsample = nn.Conv2d(in_channels=32, out_channels=self.out_channels, kernel_size=(3, 3),
                                  stride=(1, 1), padding=1, bias=True)

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)

        x = self.upscale(x)
        x = self.upsample(x)
        # print(x.shape)
        return x