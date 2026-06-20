"""
Dataset classes for all five HorusEye training tasks.

DenoiseDataset   – self-supervised denoising (pre-training)
SRDataset        – 4× super-resolution fine-tuning
MotionDataset    – motion-artifact removal fine-tuning
MetalDataset     – metal-artifact removal fine-tuning
ThicknessDataset – thick→thin slice fine-tuning

All datasets load .npy or .npz files from a directory.
Images must be pre-normalised to [0, 1] and have shape (H, W).
"""
import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, RandFlip, RandRotate90, RandAffine

from training_repro.noise_simulation import simulate_noisy_direct, simulate_noisy_sinogram

# ── Spatial augmentation shared across tasks ──────────────────────────────────

_SPATIAL_AUGMENT = Compose([
    RandAffine(
        prob=0.5,
        padding_mode="zeros",
        spatial_size=(512, 512),
        translate_range=(64, 64),
        rotate_range=(np.pi / 10, np.pi / 10),
        scale_range=(-0.4, 0.5),
    ),
    RandFlip(prob=0.5),
    RandRotate90(prob=0.5),
])


def _load_image(path: str) -> np.ndarray:
    """Load a .npy or .npz file and return a 2D float32 array."""
    if path.endswith(".npz"):
        return np.load(path)["arr_0"].astype(np.float32)
    return np.load(path).astype(np.float32)


def _collect_files(data_dir: str) -> list:
    files = (
        glob.glob(os.path.join(data_dir, "*.npy")) +
        glob.glob(os.path.join(data_dir, "*.npz"))
    )
    random.shuffle(files)
    return files


def _to_3ch_tensor(img: np.ndarray) -> torch.Tensor:
    """Stack a 2D image into a 3-channel tensor (B=1 not included)."""
    return torch.tensor(np.stack([img, img, img], axis=0), dtype=torch.float32)


def _to_1ch_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.tensor(img[np.newaxis], dtype=torch.float32)


# ── 1. Denoising (pre-training) ───────────────────────────────────────────────

class DenoiseDataset(Dataset):
    """
    Self-supervised denoising dataset.

    For each clean image, synthetic Poisson noise is applied at runtime.
    Input : noisy CT (3-channel repeated), shape (3, H, W)
    Target: clean CT (1-channel),          shape (1, H, W)
    """

    def __init__(self, data_dir: str,
                 I0: float = 1e4,
                 use_sinogram_noise: bool = False,
                 num_angles: int = 180,
                 augment: bool = True):
        self.files   = _collect_files(data_dir)
        self.I0      = I0
        self.use_sino = use_sinogram_noise
        self.num_ang  = num_angles
        self.augment  = augment
        assert len(self.files) > 0, f"No .npy/.npz files found in {data_dir}"

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        clean = _load_image(self.files[idx])
        clean = np.clip(clean, 0, 1)

        if self.augment:
            clean = _SPATIAL_AUGMENT(clean[np.newaxis])[0]

        if self.use_sino:
            noisy = simulate_noisy_sinogram(clean, self.I0, self.num_ang)
        else:
            noisy = simulate_noisy_direct(clean, self.I0)

        return _to_3ch_tensor(noisy), _to_1ch_tensor(clean)


# ── 2. Super-resolution fine-tuning ───────────────────────────────────────────

class SRDataset(Dataset):
    """
    4× super-resolution dataset.

    Input : bicubically downscaled image (3-channel repeated), shape (3, H/4, W/4)
    Target: original high-res image (1-channel),               shape (1, H, W)

    Note: the model uses SlidingWindowInferer at test time, so during training
    the downscaled image is resized back to (H, W) before entering the model.
    We replicate the paper's approach: resize LR to HR size, feed to model.
    """

    def __init__(self, data_dir: str, scale: float = 0.25, augment: bool = True):
        self.files   = _collect_files(data_dir)
        self.scale   = scale
        self.augment = augment
        assert len(self.files) > 0

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        hr = _load_image(self.files[idx])
        hr = np.clip(hr, 0, 1)

        if self.augment:
            hr = _SPATIAL_AUGMENT(hr[np.newaxis])[0]

        h, w = hr.shape
        lr = cv2.resize(hr, (int(w * self.scale), int(h * self.scale)),
                        interpolation=cv2.INTER_CUBIC)
        # Upsample LR back to HR size (bicubic) — this is the model input
        lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
        lr_up = np.clip(lr_up, 0, 1)

        return _to_3ch_tensor(lr_up), _to_1ch_tensor(hr)


# ── 3. Motion-artifact removal fine-tuning ────────────────────────────────────

class MotionDataset(Dataset):
    """
    Motion-artifact removal dataset.
    Motion artifacts are synthesised at runtime from clean images.

    Input : motion-blurred image (3-channel repeated)
    Target: clean image (1-channel)
    """

    def __init__(self, data_dir: str, augment: bool = True):
        self.files   = _collect_files(data_dir)
        self.augment = augment
        assert len(self.files) > 0

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        clean = _load_image(self.files[idx])
        clean = np.clip(clean, 0, 1)

        if self.augment:
            clean = _SPATIAL_AUGMENT(clean[np.newaxis])[0]

        # Lazy import to avoid circular dep
        from analysis.motion import generate_grid_motion, generate_nongrid_motion

        rng   = np.random
        amp   = tuple(rng.uniform(1, 5, size=(2,)))
        freq  = tuple(rng.randint(1, 3, size=(2,)).astype(float))
        trans = float(rng.uniform(1, 5))
        rot   = float(rng.uniform(0.5, 2.5))

        if rng.rand() < 0.5:
            corrupted = generate_grid_motion(clean, max_translation=trans, max_rotation=rot)
            if rng.rand() < 0.5:
                corrupted = generate_nongrid_motion(corrupted, amplitude=amp, frequency=freq)
        else:
            corrupted = generate_nongrid_motion(clean, amplitude=amp, frequency=freq)
            if rng.rand() > 0.5:
                corrupted = generate_grid_motion(corrupted, max_translation=trans, max_rotation=rot)

        corrupted = np.clip(corrupted, 0, 1).astype(np.float32)
        return _to_3ch_tensor(corrupted), _to_1ch_tensor(clean)


# ── 4. Metal-artifact removal fine-tuning ─────────────────────────────────────

class MetalDataset(Dataset):
    """
    Metal-artifact removal dataset.

    Expects .npz files where arr_0 has shape (N, H, W):
      - arr_0[-1]        : clean GT image
      - arr_0[0..N-2]    : paired metal-artifact images (one chosen randomly)

    Falls back to synthetic metal simulation when files lack paired data
    (i.e., when arr_0 is 2D).
    """

    def __init__(self, data_dir: str, augment: bool = True):
        self.files   = _collect_files(data_dir)
        self.augment = augment
        assert len(self.files) > 0

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = _load_image(self.files[idx])

        if data.ndim == 3 and data.shape[0] >= 2:
            # Real paired data: random metal slice + clean GT
            gt    = np.clip(data[-1], 0, 1)
            metal = np.clip(data[np.random.randint(0, data.shape[0] - 1)], 0, 1)
        else:
            # Synthetic: add random bright spots to simulate metal implants
            if data.ndim == 3:
                data = data[0]
            gt    = np.clip(data, 0, 1)
            metal = self._add_synthetic_metal(gt.copy())

        if self.augment:
            stacked = _SPATIAL_AUGMENT(
                np.stack([metal, gt], axis=0)
            )
            metal, gt = stacked[0], stacked[1]

        mask = np.array(metal < 0.98, dtype=np.float32)   # non-metal region

        metal_t = _to_3ch_tensor(np.clip(metal, 0, 1))
        gt_t    = _to_1ch_tensor(np.clip(gt, 0, 1))
        mask_t  = torch.tensor(mask[np.newaxis], dtype=torch.float32)
        return metal_t, gt_t, mask_t

    @staticmethod
    def _add_synthetic_metal(img: np.ndarray, n_spots: int = 3) -> np.ndarray:
        h, w = img.shape
        for _ in range(n_spots):
            cx = np.random.randint(w // 4, 3 * w // 4)
            cy = np.random.randint(h // 4, 3 * h // 4)
            r  = np.random.randint(5, 20)
            y, x = np.ogrid[:h, :w]
            disk = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
            img[disk] = 1.0
        return img


# ── 5. Slice-thickness reduction fine-tuning ──────────────────────────────────

class ThicknessDataset(Dataset):
    """
    Thick→thin slice reduction dataset.

    Expects .npz files where arr_0 has shape (C, H, W) with C slices.
    - First C//2 channels : thick-slice input
    - Last  C//2 channels : thin-slice target

    Falls back to synthetic thick-slice simulation (local averaging) when
    files contain only 2D images.
    """

    def __init__(self, data_dir: str, n_thick: int = 3, n_thin: int = 5):
        self.files   = _collect_files(data_dir)
        self.n_thick = n_thick    # input channels (thick slices)
        self.n_thin  = n_thin     # output channels (thin slices)
        assert len(self.files) > 0

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = _load_image(self.files[idx])

        if data.ndim == 3 and data.shape[0] >= self.n_thick + self.n_thin:
            thick = data[:self.n_thick].astype(np.float32)
            thin  = data[self.n_thick: self.n_thick + self.n_thin].astype(np.float32)
        else:
            # Synthetic: thin slices = single image tiled; thick = average blur
            if data.ndim == 3:
                data = data[0]
            thin  = np.stack([data] * self.n_thin,  axis=0).astype(np.float32)
            thick = np.stack(
                [np.clip(cv2.blur(data, (5 * (i + 1), 5 * (i + 1))), 0, 1)
                 for i in range(self.n_thick)],
                axis=0,
            ).astype(np.float32)

        thick = np.clip(thick, 0, 1)
        thin  = np.clip(thin,  0, 1)
        return (torch.tensor(thick, dtype=torch.float32),
                torch.tensor(thin,  dtype=torch.float32))
