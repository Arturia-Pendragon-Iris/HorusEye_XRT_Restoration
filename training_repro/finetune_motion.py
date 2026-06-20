"""
HorusEye fine-tuning: motion-artifact removal.

Motion artifacts (rigid translation/rotation + non-rigid respiratory-like
deformation) are synthesised at runtime from clean images.

Usage:
    python -m training_repro.finetune_motion \\
        --data_dir /path/to/clean_slices \\
        --pretrain_ckpt training_repro/checkpoints/pretrain_denoise/...pth
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from training_repro.config import TrainConfig, ModelConfig
from training_repro.dataset import MotionDataset
from training_repro.model import SwinUNet
from training_repro.losses import ln_loss
from training_repro.trainer import Trainer


class MotionTrainer(Trainer):
    def compute_loss(self, batch: tuple) -> torch.Tensor:
        corrupted, clean = self._to_device(batch)
        pred = self.model(corrupted)
        return ln_loss(pred, clean)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HorusEye motion-artifact removal fine-tuning")
    p.add_argument("--data_dir",      type=str,   default="")
    p.add_argument("--save_dir",      type=str,   default="training_repro/checkpoints")
    p.add_argument("--pretrain_ckpt", type=str,   default="")
    p.add_argument("--num_epochs",    type=int,   default=10)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--device",        type=str,   default="cuda")
    p.add_argument("--no_amp",        action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    cfg = TrainConfig(
        data_dir   = args.data_dir or TrainConfig().data_dir,
        save_dir   = args.save_dir,
        num_epochs = args.num_epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        device     = args.device,
        amp        = not args.no_amp,
    )

    model = SwinUNet(ModelConfig())

    if args.pretrain_ckpt:
        ckpt = torch.load(args.pretrain_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"Loaded pretrained weights from {args.pretrain_ckpt}")

    model.freeze_encoder()
    print(f"Trainable params after freeze: {model.num_trainable_params:,} "
          f"/ {model.num_params:,}")

    dataset = MotionDataset(data_dir=cfg.data_dir)
    print(f"Dataset size: {len(dataset)} images")

    trainer = MotionTrainer(model, dataset, cfg, run_name="finetune_motion")
    stats   = trainer.train()
    print(f"\nMotion fine-tuning finished: {stats}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
