"""
HorusEye fine-tuning: metal-artifact removal.

Metal regions are identified by a high-intensity mask.
Loss is computed only on non-metal voxels so that the network learns
to in-paint metal regions from surrounding context.

Usage:
    python -m training_repro.finetune_metal \\
        --data_dir /path/to/paired_metal_data \\
        --pretrain_ckpt training_repro/checkpoints/pretrain_denoise/...pth
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from training_repro.config import TrainConfig, ModelConfig
from training_repro.dataset import MetalDataset
from training_repro.model import SwinUNet
from training_repro.losses import ln_loss
from training_repro.trainer import Trainer


class MetalTrainer(Trainer):
    def compute_loss(self, batch: tuple) -> torch.Tensor:
        metal, gt, mask = self._to_device(batch)
        pred = self.model(metal)
        # Masked loss: supervise only non-metal regions
        return ln_loss(pred * mask, gt * mask)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HorusEye metal-artifact removal fine-tuning")
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

    dataset = MetalDataset(data_dir=cfg.data_dir)
    print(f"Dataset size: {len(dataset)} images")

    trainer = MetalTrainer(model, dataset, cfg, run_name="finetune_metal")
    stats   = trainer.train()
    print(f"\nMetal fine-tuning finished: {stats}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
