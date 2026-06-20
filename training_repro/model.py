"""
SwinUNet: SwinUNETR backbone + lightweight convolution head.

Architecture matches the HorusEye paper:
  - Encoder : SwinUNETR with depths=(2,2,18,2) and feature_size=96
  - Head    : two 3×3 conv layers (channel reduction → 1 channel)
  - Residual: output += central input channel (middle channel of the 3-ch input)

For fine-tuning, freeze_encoder() locks swinViT and encoder1-4;
only the decoder and conv head remain trainable.
"""
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from training_repro.config import ModelConfig, SmallModelConfig


class SwinUNet(nn.Module):
    def __init__(self, cfg: ModelConfig | SmallModelConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()

        self.in_channels  = cfg.in_channels
        self.out_channels = cfg.out_channels

        self.swin = SwinUNETR(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            depths=list(cfg.depths),
            spatial_dims=cfg.spatial_dims,
            use_checkpoint=cfg.use_checkpoint,
            feature_size=cfg.feature_size,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(cfg.out_channels, cfg.out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.out_channels, cfg.out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, do_residual: bool = True) -> torch.Tensor:
        feat = self.swin(x)
        out  = self.conv(feat)
        if do_residual:
            mid = x.shape[1] // 2      # middle channel of the 3-ch input
            out = out + x[:, mid:mid + 1]
        return out

    def freeze_encoder(self) -> None:
        """Lock backbone + skip-connection encoders; keep decoders trainable."""
        frozen = [
            self.swin.swinViT,
            self.swin.encoder1,
            self.swin.encoder2,
            self.swin.encoder3,
            self.swin.encoder4,
            self.swin.encoder10,
        ]
        for module in frozen:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
