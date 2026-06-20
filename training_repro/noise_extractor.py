"""
Stage-1: Inter-slice prediction network for self-supervised noise extraction.

The model takes two neighboring slices (z-1, z+1) as a 2-channel input and
predicts the middle slice (z).  Because noise is independently acquired, the
network learns to suppress it while leveraging structural continuity.

Noise estimate = raw_middle_slice - predicted_middle_slice.

This module provides:
  - NoiseExtractor  : SwinUNet variant with in_channels=2
  - NoiseExtractorTrainer : Trainer subclass for Stage-1
  - extract_noise_bank : run inference on a PseudoVolumeDataset and collect residuals
"""
import os
import contextlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.networks.nets import SwinUNETR

from training_repro.config import ModelConfig, SmallModelConfig, TrainConfig
from training_repro.losses import ln_loss, sharp_loss
from training_repro.trainer import Trainer
from training_repro.inter_slice_dataset import PseudoVolumeDataset


class NoiseExtractor(nn.Module):
    """
    SwinUNet variant for inter-slice prediction.

    in_channels = 2  : (slice_z-1, slice_z+1)
    out_channels = 1 : predicted middle slice

    No residual skip — we need the absolute prediction, not a delta.
    """

    def __init__(self, cfg: ModelConfig | SmallModelConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()

        extractor_cfg = type(cfg)(
            in_channels=2,
            out_channels=1,
            feature_size=cfg.feature_size,
            depths=cfg.depths,
            spatial_dims=cfg.spatial_dims,
            use_checkpoint=cfg.use_checkpoint,
        )

        self.swin = SwinUNETR(
            in_channels=2,
            out_channels=1,
            depths=list(extractor_cfg.depths),
            spatial_dims=extractor_cfg.spatial_dims,
            use_checkpoint=extractor_cfg.use_checkpoint,
            feature_size=extractor_cfg.feature_size,
        )
        self.head = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, H, W)  — left and right neighboring slices
        Returns:
            pred: (B, 1, H, W) — predicted middle slice
        """
        feat = self.swin(x)
        return self.head(feat)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class NoiseExtractorTrainer(Trainer):
    """
    Trains NoiseExtractor (inter-slice prediction).

    Loss = reconstruction_loss(predicted_middle, raw_middle)
         = ln_loss + lambda_sharp * sharp_loss
    """

    def compute_loss(self, batch: tuple) -> torch.Tensor:
        inp, target = self._to_device(batch)   # (B,2,H,W), (B,1,H,W)
        pred = self.model(inp)
        loss = ln_loss(pred, target)
        loss = loss + self.cfg.lambda_sharp * sharp_loss(pred, target)
        return loss


def extract_noise_bank(
    model: NoiseExtractor,
    data_dir: str,
    device: torch.device,
    batch_size: int = 4,
    num_workers: int = 0,
) -> list[np.ndarray]:
    """
    Run the trained noise extractor over every valid triplet and collect
    residuals:  noise_i = clip(raw_middle_i - predicted_middle_i)

    Returns a list of 2D numpy float32 arrays (the noise samples).
    """
    dataset = PseudoVolumeDataset(data_dir)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    model.eval()
    noise_bank: list[np.ndarray] = []
    idx = 0

    with torch.no_grad():
        for inp, target in loader:
            inp    = inp.to(device)
            target = target.to(device)
            pred   = model(inp)                        # (B,1,H,W)
            residual = (target - pred).cpu().numpy()   # positive or negative

            for b in range(residual.shape[0]):
                raw_noise = residual[b, 0]             # (H, W)
                noise_bank.append(raw_noise.astype(np.float32))
            idx += inp.shape[0]

    print(f"  extracted {len(noise_bank)} noise samples from {data_dir}")
    return noise_bank
