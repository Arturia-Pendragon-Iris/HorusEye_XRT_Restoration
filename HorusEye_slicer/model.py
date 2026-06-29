import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNet(nn.Module):
    def __init__(self, in_ch=3, feature_ch=96, final_ch=1):
        super().__init__()
        swin_args = dict(
            in_channels=in_ch,
            out_channels=final_ch,
            depths=[2, 2, 18, 2],
            spatial_dims=2,
            use_checkpoint=True,
            feature_size=feature_ch,
        )
        try:
            self.swin = SwinUNETR(img_size=(512, 512), **swin_args)
        except TypeError:
            self.swin = SwinUNETR(**swin_args)
        self.conv = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(final_ch, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, do_residual=True):
        residual = x
        features = self.swin(x)
        output = self.conv(features)
        if do_residual:
            return output + residual[:, 1:2]
        return output
