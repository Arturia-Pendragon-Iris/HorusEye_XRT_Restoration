import itertools
from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR


class SwinUNet(nn.Module):
    def __init__(self, feature_ch=96, final_ch=3):
        super(SwinUNet, self).__init__()
        self.swin = SwinUNETR(img_size=(512, 512),
                              in_channels=3,
                              out_channels=final_ch,
                              depths=[2, 2, 18, 2],
                              spatial_dims=2,
                              use_checkpoint=True,
                              feature_size=feature_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(final_ch, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        residual = x[:, 1]
        F = self.swin(x)
        out = self.conv(F)
        return out + residual
