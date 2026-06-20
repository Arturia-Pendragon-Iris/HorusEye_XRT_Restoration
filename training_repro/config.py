"""
HorusEye training configuration.
ModelConfig controls the SwinUNet architecture.
TrainConfig controls training hyperparameters and paths.
"""
import os
from dataclasses import dataclass, field
from typing import Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ModelConfig:
    in_channels: int = 3
    out_channels: int = 1
    feature_size: int = 96          # paper uses 96
    depths: Tuple[int, ...] = (2, 2, 18, 2)   # paper uses (2,2,18,2)
    spatial_dims: int = 2
    use_checkpoint: bool = True


@dataclass
class SmallModelConfig:
    """Lightweight config for quick tests / CI."""
    in_channels: int = 3
    out_channels: int = 1
    feature_size: int = 24
    depths: Tuple[int, ...] = (2, 2, 2, 2)
    spatial_dims: int = 2
    use_checkpoint: bool = False


@dataclass
class TrainConfig:
    # ── paths ────────────────────────────────────────────────────────────────
    data_dir: str = os.path.join(PROJECT_ROOT, "example_dataset")
    save_dir: str = os.path.join(PROJECT_ROOT, "training_repro", "checkpoints")
    pretrain_ckpt: str = ""          # path to base checkpoint for fine-tuning

    # ── training ─────────────────────────────────────────────────────────────
    batch_size: int = 2
    num_epochs: int = 10
    lr: float = 1e-4
    lr_step: int = 2                 # StepLR epoch step
    lr_gamma: float = 0.8
    num_workers: int = 0             # 0 for Windows (no fork)
    seed: int = 42
    device: str = "cuda"
    amp: bool = True                 # automatic mixed precision

    # ── loss ─────────────────────────────────────────────────────────────────
    lambda_sharp: float = 0.5        # weight for sharpness loss

    # ── noise simulation (pre-training) ──────────────────────────────────────
    I0: float = 1e4                  # incident photon count; lower → noisier
    num_angles: int = 180            # projection angles for sinogram simulation

    # ── SR fine-tuning ────────────────────────────────────────────────────────
    sr_scale: float = 0.25           # downscale factor (4× SR)

    # ── logging ──────────────────────────────────────────────────────────────
    log_interval: int = 10           # print loss every N iterations
    save_interval: int = 2           # save checkpoint every N epochs
