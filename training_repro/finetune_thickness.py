"""
HorusEye fine-tuning: slice-thickness reduction (thick → thin).

Input : 3 thick-slice images (3-channel input)
Output: 5 thin-slice images  (5-channel output)

Loss: reconstruction_loss on all 5 output slices
      + flow_loss to enforce inter-slice consistency

Usage:
    python -m training_repro.finetune_thickness \\
        --data_dir /path/to/5mm_data \\
        --pretrain_ckpt training_repro/checkpoints/pretrain_denoise/...pth
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from training_repro.config import TrainConfig, ModelConfig, SmallModelConfig
from training_repro.dataset import ThicknessDataset
from training_repro.model import SwinUNet
from training_repro.losses import reconstruction_loss, flow_loss
from training_repro.trainer import Trainer


class ThicknessModelConfig(ModelConfig):
    """5-output-channel variant for thickness task."""
    out_channels: int = 5


class ThicknessTrainer(Trainer):
    def compute_loss(self, batch: tuple) -> torch.Tensor:
        thick, thin = self._to_device(batch)
        pred = self.model(thick, do_residual=False)
        loss = reconstruction_loss(pred, thin, self.cfg.lambda_sharp)
        loss = loss + flow_loss(pred, thin)
        return loss


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HorusEye thickness-reduction fine-tuning")
    p.add_argument("--data_dir",      type=str,   default="")
    p.add_argument("--save_dir",      type=str,   default="training_repro/checkpoints")
    p.add_argument("--pretrain_ckpt", type=str,   default="")
    p.add_argument("--num_epochs",    type=int,   default=10)
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--lambda_sharp",  type=float, default=0.5)
    p.add_argument("--device",        type=str,   default="cuda")
    p.add_argument("--no_amp",        action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    cfg = TrainConfig(
        data_dir     = args.data_dir or TrainConfig().data_dir,
        save_dir     = args.save_dir,
        num_epochs   = args.num_epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        lambda_sharp = args.lambda_sharp,
        device       = args.device,
        amp          = not args.no_amp,
    )

    # Thickness task uses 3-in / 5-out model
    model_cfg = ThicknessModelConfig()
    model = SwinUNet(model_cfg)

    if args.pretrain_ckpt:
        ckpt = torch.load(args.pretrain_ckpt, map_location="cpu")
        # Load only matching keys (encoder/decoder; skip output head which changed shape)
        state = ckpt["model"]
        model_state = model.state_dict()
        state = {k: v for k, v in state.items()
                 if k in model_state and v.shape == model_state[k].shape}
        model_state.update(state)
        model.load_state_dict(model_state)
        print(f"Loaded {len(state)} matching keys from {args.pretrain_ckpt}")

    model.freeze_encoder()
    print(f"Trainable params after freeze: {model.num_trainable_params:,} "
          f"/ {model.num_params:,}")

    dataset = ThicknessDataset(data_dir=cfg.data_dir, n_thick=3, n_thin=5)
    print(f"Dataset size: {len(dataset)} images")

    trainer = ThicknessTrainer(model, dataset, cfg, run_name="finetune_thickness")
    stats   = trainer.train()
    print(f"\nThickness fine-tuning finished: {stats}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
