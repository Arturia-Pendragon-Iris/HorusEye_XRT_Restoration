"""
HorusEye self-supervised pre-training: CT denoising.

Usage (from project root):
    python -m training_repro.pretrain_denoise \\
        --data_dir /path/to/clean_ct_slices \\
        --save_dir training_repro/checkpoints \\
        --num_epochs 10 \\
        --batch_size 4

The model trains to recover clean CT images from synthetically corrupted
inputs produced by log-Poisson noise simulation.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from training_repro.config import TrainConfig, ModelConfig
from training_repro.dataset import DenoiseDataset
from training_repro.model import SwinUNet
from training_repro.losses import reconstruction_loss
from training_repro.trainer import Trainer


class DenoiseTrainer(Trainer):
    def compute_loss(self, batch: tuple) -> torch.Tensor:
        noisy, clean = self._to_device(batch)
        pred = self.model(noisy)
        return reconstruction_loss(pred, clean, self.cfg.lambda_sharp)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HorusEye denoising pre-training")
    p.add_argument("--data_dir",    type=str, default="")
    p.add_argument("--save_dir",    type=str, default="training_repro/checkpoints")
    p.add_argument("--num_epochs",  type=int, default=10)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--I0",          type=float, default=1e4)
    p.add_argument("--use_sinogram",action="store_true",
                   help="Use skimage Radon-based noise (slower but more accurate)")
    p.add_argument("--lambda_sharp",type=float, default=0.5)
    p.add_argument("--resume",      type=str,   default="")
    p.add_argument("--device",      type=str,   default="cuda")
    p.add_argument("--no_amp",      action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    cfg = TrainConfig(
        data_dir     = args.data_dir or TrainConfig().data_dir,
        save_dir     = args.save_dir,
        num_epochs   = args.num_epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        I0           = args.I0,
        lambda_sharp = args.lambda_sharp,
        device       = args.device,
        amp          = not args.no_amp,
    )

    model_cfg = ModelConfig()
    model     = SwinUNet(model_cfg)
    print(f"Model total params    : {model.num_params:,}")
    print(f"Model trainable params: {model.num_trainable_params:,}")

    dataset = DenoiseDataset(
        data_dir            = cfg.data_dir,
        I0                  = cfg.I0,
        use_sinogram_noise  = args.use_sinogram,
        num_angles          = cfg.num_angles,
    )
    print(f"Dataset size: {len(dataset)} images")

    trainer = DenoiseTrainer(model, dataset, cfg, run_name="pretrain_denoise")

    if args.resume:
        trainer.load_checkpoint(args.resume)

    stats = trainer.train()
    print(f"\nPre-training finished: {stats}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
