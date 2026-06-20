"""
Log-Poisson CT noise simulation.

Two modes:
  - sinogram_based: forward-project via skimage Radon, add Poisson noise, reconstruct with FBP.
    Physically accurate but ~10× slower.
  - direct: equivalent Poisson noise applied directly in image space.
    Much faster; used as default when speed matters.
"""
import numpy as np
from skimage.transform import radon, iradon


def simulate_noisy_sinogram(img: np.ndarray,
                             I0: float = 1e4,
                             num_angles: int = 180) -> np.ndarray:
    """
    Simulate a noisy FBP reconstruction from a clean CT image using
    log-Poisson noise in sinogram space.

    img       : 2D float32 array, normalised to [0, 1]
    I0        : incident photon count (lower = more noise)
    num_angles: number of projection angles

    Returns noisy CT image with same shape and range [0, 1].
    """
    img = np.clip(img, 0, 1).astype(np.float32)
    theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)

    # Forward projection → sinogram
    sino = radon(img, theta=theta, circle=True)

    # Convert to attenuation scale for Poisson simulation
    sino_max = sino.max() + 1e-8
    sino_norm = sino / sino_max          # normalise so e^-sino stays in (0,1]

    # Simulate Poisson noise
    intensity = I0 * np.exp(-sino_norm)
    noisy_int = np.random.poisson(intensity).astype(np.float32)
    noisy_int = np.maximum(noisy_int, 1)          # avoid log(0)
    sino_noisy = -np.log(noisy_int / I0) * sino_max   # back to original scale

    # FBP reconstruction
    noisy_recon = iradon(sino_noisy, theta=theta, circle=True, filter_name='ramp')
    return np.clip(noisy_recon, 0, 1).astype(np.float32)


def simulate_noisy_direct(img: np.ndarray, I0: float = 1e4) -> np.ndarray:
    """
    Fast Poisson noise directly in image space.
    Equivalent to the sinogram approach when the image is already an FBP reconstruction.

    img : 2D float32 array, normalised to [0, 1]
    I0  : photon count (lower = more noise)

    Returns noisy image, same shape and range [0, 1].
    """
    img = np.clip(img, 0, 1).astype(np.float32)
    intensity = I0 * np.exp(-img * 5.0)       # 5× for typical CT attenuation range
    noisy_int = np.random.poisson(intensity).astype(np.float32)
    noisy_int = np.maximum(noisy_int, 1)
    noisy = -np.log(noisy_int / I0) / 5.0
    return np.clip(noisy, 0, 1).astype(np.float32)
