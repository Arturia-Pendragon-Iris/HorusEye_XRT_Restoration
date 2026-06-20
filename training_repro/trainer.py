"""
Generic Trainer with:
  - AMP (automatic mixed precision)
  - StepLR scheduler
  - Checkpoint save / resume
  - Per-iteration and per-epoch logging
"""
import os
import time
import random
import contextlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR

from training_repro.config import TrainConfig


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """
    Generic single-task trainer.

    Subclass and override `compute_loss(batch)` to implement task-specific logic.
    `compute_loss` receives the raw batch tuple and must return a scalar loss tensor.
    """

    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
                 cfg: TrainConfig,
                 run_name: str = "run"):
        _set_seed(cfg.seed)

        self.cfg      = cfg
        self.run_name = run_name
        self.device   = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model    = model.to(self.device)

        self.loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
        )
        self.scheduler = StepLR(self.optimizer, step_size=cfg.lr_step, gamma=cfg.lr_gamma)
        _use_amp = cfg.amp and self.device.type == "cuda"
        self.scaler    = torch.amp.GradScaler("cuda", enabled=_use_amp)
        self._use_amp  = _use_amp

        os.makedirs(cfg.save_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(cfg.save_dir, run_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self._global_step = 0
        self._best_loss   = float("inf")

    # ── Override in subclasses ─────────────────────────────────────────────────
    def compute_loss(self, batch: tuple) -> torch.Tensor:
        raise NotImplementedError

    # ── Training loop ──────────────────────────────────────────────────────────
    def train(self, max_steps: int | None = None) -> dict:
        """
        Train for cfg.num_epochs, or stop after max_steps total iterations
        (useful for quick tests).

        Returns a dict with the final training statistics.
        """
        stats = {"epochs": 0, "steps": 0, "loss": float("inf")}
        t0 = time.time()

        for epoch in range(1, self.cfg.num_epochs + 1):
            epoch_loss = 0.0
            n_batches  = 0

            self.model.train()
            for batch in self.loader:
                _ctx = (torch.amp.autocast(self.device.type)
                        if self._use_amp else contextlib.nullcontext())
                with _ctx:
                    loss = self.compute_loss(batch)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss       += loss.item()
                n_batches        += 1
                self._global_step += 1

                if self._global_step % self.cfg.log_interval == 0:
                    avg = epoch_loss / n_batches
                    elapsed = time.time() - t0
                    print(
                        f"[{self.run_name}] epoch {epoch:3d} | "
                        f"step {self._global_step:6d} | "
                        f"loss {loss.item():.5f} | avg {avg:.5f} | "
                        f"lr {self.optimizer.param_groups[0]['lr']:.2e} | "
                        f"{elapsed:.0f}s"
                    )

                if max_steps and self._global_step >= max_steps:
                    stats.update({"epochs": epoch, "steps": self._global_step,
                                  "loss": epoch_loss / max(n_batches, 1)})
                    return stats

            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            stats.update({"epochs": epoch, "steps": self._global_step,
                          "loss": avg_epoch_loss})

            print(
                f"[{self.run_name}] ── epoch {epoch} done | "
                f"avg_loss {avg_epoch_loss:.5f}"
            )

            self.scheduler.step()

            if epoch % self.cfg.save_interval == 0:
                self.save_checkpoint(epoch, avg_epoch_loss)

        return stats

    # ── Checkpoint helpers ─────────────────────────────────────────────────────
    def save_checkpoint(self, epoch: int, loss: float) -> None:
        path = os.path.join(self.ckpt_dir, f"epoch_{epoch:04d}_loss_{loss:.5f}.pth")
        torch.save({
            "epoch":      epoch,
            "loss":       loss,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
        }, path)
        print(f"  checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        epoch = ckpt.get("epoch", 0)
        print(f"  loaded checkpoint from epoch {epoch}: {path}")
        return epoch

    def _to_device(self, batch: tuple) -> tuple:
        return tuple(
            x.to(self.device) if isinstance(x, torch.Tensor) else x
            for x in batch
        )
