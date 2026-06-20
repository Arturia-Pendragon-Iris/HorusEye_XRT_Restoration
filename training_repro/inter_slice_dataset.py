"""
Inter-slice prediction dataset for Stage-1 noise extraction.

The key insight from the paper:
  - Adjacent slices share structural continuity.
  - Noise is acquired independently per slice (no inter-slice correlation).
  => Predicting the middle slice from its two neighbors suppresses noise
     while preserving anatomy, and the residual  (raw_z - predicted_z)
     approximates the per-acquisition noise.

Data formats accepted:
  A)  .npz files with shape (N, H, W)  — a single CT volume per file.
  B)  Directory of sorted .npy/.npz 2D slices — treated as one pseudo-volume.
      Useful for testing with example_dataset/.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_2d(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        arr = np.load(path)["arr_0"]
    else:
        arr = np.load(path)
    arr = arr.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]          # take first slice if 3D
    return np.clip(arr, 0, 1)


class InterSliceVolumeDataset(Dataset):
    """
    Triplet dataset for inter-slice prediction.

    Each sample is  (left, right) → target_middle,
    where left  = slice z-1,
          right = slice z+1,
          middle= slice z   (noisy reference).

    `volume_paths`: list of .npz volume files (shape [N, H, W]).
    """

    def __init__(self, volume_paths: list[str]):
        self.triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for path in volume_paths:
            vol = np.load(path)["arr_0"].astype(np.float32)
            vol = np.clip(vol, 0, 1)
            n = vol.shape[0]
            for z in range(1, n - 1):
                self.triplets.append((vol[z - 1], vol[z + 1], vol[z]))

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        left, right, mid = self.triplets[idx]
        inp = torch.tensor(
            np.stack([left, right], axis=0), dtype=torch.float32
        )  # (2, H, W)
        target = torch.tensor(mid[np.newaxis], dtype=torch.float32)   # (1, H, W)
        return inp, target


class PseudoVolumeDataset(Dataset):
    """
    Treats a directory of sorted 2D .npy/.npz slices as a single volume.
    Consecutive files become neighboring slices.
    Useful when you only have 2D slice files (e.g., example_dataset/).
    """

    def __init__(self, data_dir: str):
        files = sorted(
            glob.glob(os.path.join(data_dir, "*.npy")) +
            glob.glob(os.path.join(data_dir, "*.npz"))
        )
        assert len(files) >= 3, (
            f"Need at least 3 slice files for inter-slice triplets, "
            f"found {len(files)} in {data_dir}"
        )
        self.slices = [_load_2d(f) for f in files]

    def __len__(self) -> int:
        return len(self.slices) - 2          # valid range: 1 … N-2

    def __getitem__(self, idx: int):
        left  = self.slices[idx]             # z-1
        mid   = self.slices[idx + 1]         # z
        right = self.slices[idx + 2]         # z+1

        inp    = torch.tensor(
            np.stack([left, right], axis=0), dtype=torch.float32
        )  # (2, H, W)
        target = torch.tensor(mid[np.newaxis], dtype=torch.float32)
        return inp, target

    def get_raw_slice(self, idx: int) -> np.ndarray:
        """Return the raw middle slice (z) as numpy array."""
        return self.slices[idx + 1]
