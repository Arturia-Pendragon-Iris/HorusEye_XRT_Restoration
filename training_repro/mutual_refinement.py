"""
Mutual Positive-Feedback Co-Refinement Pretraining  (Section 3.2 of the paper).

Two models co-evolve across T iterations:

    Iteration t:
    ┌────────────────────────────────────────────────────────────────────┐
    │  Phase A — Train NoiseExtractor on triplets, using clean_bank      │
    │            as implicit supervision (structure is already cleaner).  │
    │                                                                    │
    │  Phase B — Extract residuals:                                      │
    │            noise_bank ← { raw_z - predicted_z  for all z }        │
    │                                                                    │
    │  Phase C — Build synthetic pairs:                                  │
    │            noisy_input ← clean_bank[i]  +  noise_bank[j]          │
    │            Train Denoiser(noisy_input) → clean_bank[i]            │
    │                                                                    │
    │  Phase D — Update clean_bank with high-confidence denoiser output: │
    │            for each raw slice:                                     │
    │              out = denoiser(raw_slice + extracted_noise)           │
    │              if L1_improvement > threshold: clean_bank.add(out)   │
    └────────────────────────────────────────────────────────────────────┘

At t=0  clean_bank is seeded with the raw (already reasonably clean)
example images — in real pretraining it could be empty and the
extractor begins by pure self-supervised triplet learning.
"""
import os
import random
import glob
import contextlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from training_repro.config import TrainConfig, SmallModelConfig
from training_repro.model import SwinUNet
from training_repro.losses import ln_loss, sharp_loss
from training_repro.noise_extractor import NoiseExtractor, NoiseExtractorTrainer, extract_noise_bank
from training_repro.inter_slice_dataset import PseudoVolumeDataset


# ─────────────────────────────────────────────────────────────────────────────
# Banks
# ─────────────────────────────────────────────────────────────────────────────

class NoiseSampleBank:
    """Holds extracted noise patterns from the inter-slice residuals."""

    def __init__(self):
        self._bank: list[np.ndarray] = []

    def update(self, noise_list: list[np.ndarray]) -> None:
        self._bank.extend(noise_list)
        print(f"  NoiseSampleBank: {len(self._bank)} samples total")

    def sample(self, n: int) -> list[np.ndarray]:
        if not self._bank:
            return []
        return random.choices(self._bank, k=n)

    def __len__(self) -> int:
        return len(self._bank)


class CleanImageBank:
    """
    Stores pseudo-clean 2D images (float32, H×W, range [0,1]).

    Initially seeded from the raw slice files; high-confidence denoiser
    outputs replace entries over iterations.
    """

    def __init__(self, data_dir: str):
        files = sorted(
            glob.glob(os.path.join(data_dir, "*.npy")) +
            glob.glob(os.path.join(data_dir, "*.npz"))
        )
        self._bank: list[np.ndarray] = []
        for f in files:
            if f.endswith(".npz"):
                arr = np.load(f)["arr_0"]
            else:
                arr = np.load(f)
            arr = arr.astype(np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            self._bank.append(np.clip(arr, 0.0, 1.0))
        print(f"  CleanImageBank seeded with {len(self._bank)} raw slices")

    def update(self, idx: int, refined: np.ndarray) -> None:
        self._bank[idx] = np.clip(refined.astype(np.float32), 0.0, 1.0)

    def sample(self, n: int) -> list[np.ndarray]:
        if not self._bank:
            return []
        return random.choices(self._bank, k=n)

    def __len__(self) -> int:
        return len(self._bank)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset built from the two banks
# ─────────────────────────────────────────────────────────────────────────────

class BankDenoiseDataset(Dataset):
    """
    Returns (noisy_3ch, clean_1ch) pairs built on-the-fly from the banks.

    noisy = clean_from_bank + noise_from_bank    (clipped to [0,1])
    """

    def __init__(
        self,
        clean_bank: CleanImageBank,
        noise_bank: NoiseSampleBank,
        n_samples: int = 256,
        fallback_I0: float = 1e4,
    ):
        self.clean_bank  = clean_bank
        self.noise_bank  = noise_bank
        self.n_samples   = n_samples
        self.fallback_I0 = fallback_I0

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        clean = self.clean_bank.sample(1)[0]   # (H, W)

        if len(self.noise_bank) > 0:
            noise = self.noise_bank.sample(1)[0]  # (H, W)
            if noise.shape != clean.shape:
                # size mismatch: fall back to synthetic Poisson noise
                noise = _synthetic_noise(clean, self.fallback_I0)
            noisy = np.clip(clean + noise, 0.0, 1.0)
        else:
            noisy = _synthetic_noise(clean, self.fallback_I0)
            noisy = np.clip(noisy, 0.0, 1.0)

        noisy_3ch = np.stack([noisy, noisy, noisy], axis=0)   # (3, H, W)
        clean_1ch = clean[np.newaxis]                          # (1, H, W)

        return (
            torch.tensor(noisy_3ch, dtype=torch.float32),
            torch.tensor(clean_1ch, dtype=torch.float32),
        )


def _synthetic_noise(clean: np.ndarray, I0: float) -> np.ndarray:
    """Quick image-space log-Poisson noise fallback."""
    attn  = -np.log(np.clip(clean, 1e-6, 1.0))
    noisy = np.random.poisson(I0 * np.exp(-attn)).astype(np.float32)
    noisy = -np.log(np.clip(noisy / I0, 1e-6, 1.0))
    noisy = noisy / (-np.log(1e-6))
    return np.clip(noisy, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Denoiser trainer (Stage 2, bank-driven)
# ─────────────────────────────────────────────────────────────────────────────

class BankDenoiseTrainer:
    """
    Trains the main denoiser using (clean_bank, noise_bank) synthetic pairs.
    Lightweight trainer; not inheriting from Trainer to keep it self-contained.
    """

    def __init__(
        self,
        model: SwinUNet,
        clean_bank: CleanImageBank,
        noise_bank: NoiseSampleBank,
        cfg: TrainConfig,
    ):
        _use_amp = cfg.amp and torch.cuda.is_available()
        self.device  = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model   = model.to(self.device)
        self.cfg     = cfg
        self._use_amp = _use_amp

        self.optim  = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=_use_amp)

        self.clean_bank = clean_bank
        self.noise_bank = noise_bank

    def train_steps(self, n_steps: int, batch_size: int = 4) -> float:
        dataset = BankDenoiseDataset(self.clean_bank, self.noise_bank)
        loader  = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        self.model.train()
        total_loss = 0.0
        step = 0
        loader_iter = iter(loader)

        while step < n_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            noisy, clean = batch
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            _ctx = (torch.amp.autocast(self.device.type)
                    if self._use_amp else contextlib.nullcontext())
            with _ctx:
                pred  = self.model(noisy)
                loss  = ln_loss(pred, clean)
                loss += self.cfg.lambda_sharp * sharp_loss(pred, clean)

            self.optim.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

            total_loss += loss.item()
            step += 1

        return total_loss / max(step, 1)

    @torch.no_grad()
    def update_clean_bank(
        self,
        clean_bank: CleanImageBank,
        improvement_threshold: float = 0.005,
    ) -> int:
        """
        Run denoiser on each raw slice + some noise; if improvement > threshold,
        update the clean bank entry with the denoised output.

        Returns the number of entries updated.
        """
        self.model.eval()
        updated = 0

        for idx, raw in enumerate(clean_bank._bank):
            # Build a noisy version to feed to the denoiser
            if len(self.noise_bank) > 0:
                noise = self.noise_bank.sample(1)[0]
                if noise.shape == raw.shape:
                    noisy = np.clip(raw + noise, 0.0, 1.0)
                else:
                    noisy = _synthetic_noise(raw, self.cfg.I0)
            else:
                noisy = _synthetic_noise(raw, self.cfg.I0)

            noisy_t = torch.tensor(
                np.stack([noisy, noisy, noisy], axis=0)[np.newaxis],
                dtype=torch.float32,
                device=self.device,
            )  # (1, 3, H, W)

            out = self.model(noisy_t)                 # (1, 1, H, W)
            denoised = out[0, 0].cpu().numpy()

            # High confidence: L1 improvement over the raw slice
            improvement = np.mean(np.abs(raw - noisy)) - np.mean(np.abs(raw - denoised))
            if improvement > improvement_threshold:
                clean_bank.update(idx, denoised)
                updated += 1

        print(f"  CleanImageBank: {updated}/{len(clean_bank)} entries updated")
        return updated


# ─────────────────────────────────────────────────────────────────────────────
# Main co-refinement orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class MutualRefinementPretrain:
    """
    Full two-stage co-refinement pretraining pipeline.

    Usage:
        pipeline = MutualRefinementPretrain(
            noise_extractor_cfg=SmallModelConfig(),
            denoiser_cfg=SmallModelConfig(),
            data_dir="example_dataset",
            cfg=train_cfg,
        )
        pipeline.run(n_iterations=3, steps_per_phase=20)
        pipeline.save("checkpoints/pretrain_mutual/")
    """

    def __init__(
        self,
        noise_extractor_cfg,
        denoiser_cfg,
        data_dir: str,
        cfg: TrainConfig,
    ):
        self.data_dir  = data_dir
        self.cfg       = cfg
        self.device    = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # Two networks
        self.noise_extractor = NoiseExtractor(noise_extractor_cfg).to(self.device)
        self.denoiser        = SwinUNet(denoiser_cfg).to(self.device)

        # Banks
        self.clean_bank = CleanImageBank(data_dir)
        self.noise_bank = NoiseSampleBank()

        # Trainers (created fresh each iteration so their optimizers see current banks)
        self._extractor_cfg = noise_extractor_cfg
        self._denoiser_cfg  = denoiser_cfg

        self._iteration = 0

    def run(
        self,
        n_iterations: int = 3,
        steps_per_phase: int = 100,
        improvement_threshold: float = 0.005,
    ) -> dict:
        """
        Run `n_iterations` of the mutual co-refinement loop.

        Each iteration executes four phases (A → B → C → D).
        Returns a summary dict with per-iteration losses.
        """
        history = []

        for t in range(n_iterations):
            self._iteration += 1
            print(f"\n{'='*60}")
            print(f"  Co-Refinement Iteration {self._iteration}")
            print(f"{'='*60}")

            # ── Phase A: Train NoiseExtractor ──────────────────────────────
            print("\n  Phase A — Training NoiseExtractor (inter-slice prediction)")
            extractor_trainer = NoiseExtractorTrainer(
                model   = self.noise_extractor,
                dataset = PseudoVolumeDataset(self.data_dir),
                cfg     = self.cfg,
                run_name= f"noise_extractor_iter{self._iteration}",
            )
            extractor_stats = extractor_trainer.train(max_steps=steps_per_phase)
            extractor_loss  = extractor_stats["loss"]
            print(f"  Phase A done — extractor loss: {extractor_loss:.5f}")

            # ── Phase B: Extract noise residuals ───────────────────────────
            print("\n  Phase B — Extracting noise residuals")
            new_noise = extract_noise_bank(
                model      = self.noise_extractor,
                data_dir   = self.data_dir,
                device     = self.device,
                batch_size = self.cfg.batch_size,
                num_workers= self.cfg.num_workers,
            )
            self.noise_bank.update(new_noise)

            # ── Phase C: Train Denoiser with extracted noise ────────────────
            print("\n  Phase C — Training Denoiser (bank-driven)")
            denoiser_trainer = BankDenoiseTrainer(
                model      = self.denoiser,
                clean_bank = self.clean_bank,
                noise_bank = self.noise_bank,
                cfg        = self.cfg,
            )
            denoiser_loss = denoiser_trainer.train_steps(
                n_steps   = steps_per_phase,
                batch_size= self.cfg.batch_size,
            )
            print(f"  Phase C done — denoiser loss: {denoiser_loss:.5f}")

            # ── Phase D: Update clean bank with high-confidence outputs ─────
            print("\n  Phase D — Updating clean bank from denoiser outputs")
            n_updated = denoiser_trainer.update_clean_bank(
                self.clean_bank,
                improvement_threshold=improvement_threshold,
            )

            history.append({
                "iteration":      self._iteration,
                "extractor_loss": extractor_loss,
                "denoiser_loss":  denoiser_loss,
                "clean_updated":  n_updated,
                "noise_bank_size":len(self.noise_bank),
            })

        print("\n  Co-refinement complete.")
        for h in history:
            print(
                f"  iter {h['iteration']}: "
                f"extractor={h['extractor_loss']:.4f}  "
                f"denoiser={h['denoiser_loss']:.4f}  "
                f"clean_updated={h['clean_updated']}  "
                f"noise_pool={h['noise_bank_size']}"
            )
        return {"history": history}

    def save(self, save_dir: str) -> None:
        """Save both model checkpoints."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            self.noise_extractor.state_dict(),
            os.path.join(save_dir, "noise_extractor.pth"),
        )
        torch.save(
            {"model": self.denoiser.state_dict(), "epoch": self._iteration},
            os.path.join(save_dir, "denoiser.pth"),
        )
        print(f"  saved noise_extractor.pth + denoiser.pth → {save_dir}")

    def load_denoiser(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt["model"] if "model" in ckpt else ckpt
        self.denoiser.load_state_dict(state)
        print(f"  loaded denoiser from {path}")
