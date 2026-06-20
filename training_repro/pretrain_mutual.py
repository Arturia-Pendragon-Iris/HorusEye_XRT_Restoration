"""
Full HorusEye two-stage mutual co-refinement pretraining.

Stages:
  1. NoiseExtractor (inter-slice prediction, in_channels=2)
  2. Denoiser (SwinUNet, in_channels=3)
  Iterated for --n_iterations rounds.

Usage:
    python -m training_repro.pretrain_mutual \\
        --data_dir example_dataset \\
        --save_dir training_repro/checkpoints/pretrain_mutual \\
        --n_iterations 3 \\
        --steps_per_phase 500
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from training_repro.config import TrainConfig, ModelConfig
from training_repro.mutual_refinement import MutualRefinementPretrain


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HorusEye mutual co-refinement pretraining")
    p.add_argument("--data_dir",        type=str,   default="example_dataset")
    p.add_argument("--save_dir",        type=str,   default="training_repro/checkpoints/pretrain_mutual")
    p.add_argument("--n_iterations",    type=int,   default=3,    help="co-refinement iterations")
    p.add_argument("--steps_per_phase", type=int,   default=500,  help="gradient steps per phase")
    p.add_argument("--batch_size",      type=int,   default=2)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--lambda_sharp",    type=float, default=0.5)
    p.add_argument("--improvement_thr", type=float, default=0.005,
                   help="L1 improvement threshold for clean-bank update")
    p.add_argument("--device",          type=str,   default="cuda")
    p.add_argument("--no_amp",          action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    cfg = TrainConfig(
        data_dir     = args.data_dir,
        save_dir     = args.save_dir,
        batch_size   = args.batch_size,
        lr           = args.lr,
        lambda_sharp = args.lambda_sharp,
        device       = args.device,
        amp          = not args.no_amp,
    )

    model_cfg = ModelConfig()

    print("HorusEye — Mutual Co-Refinement Pretraining")
    print(f"  data_dir       : {args.data_dir}")
    print(f"  n_iterations   : {args.n_iterations}")
    print(f"  steps_per_phase: {args.steps_per_phase}")
    print(f"  device         : {cfg.device} (cuda={torch.cuda.is_available()})")

    pipeline = MutualRefinementPretrain(
        noise_extractor_cfg = model_cfg,
        denoiser_cfg        = model_cfg,
        data_dir            = args.data_dir,
        cfg                 = cfg,
    )

    result = pipeline.run(
        n_iterations         = args.n_iterations,
        steps_per_phase      = args.steps_per_phase,
        improvement_threshold= args.improvement_thr,
    )

    pipeline.save(args.save_dir)
    print(f"\nPretraining complete.  Checkpoints saved to {args.save_dir}")
    print(result)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
