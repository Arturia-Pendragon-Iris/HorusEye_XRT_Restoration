"""DICOM CT denoising runner for the 3D Slicer demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Denoise one DICOM CT slice and write display-ready outputs.")
    parser.add_argument("--input-dicom", required=True, type=Path, help="Input DICOM file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for denoised outputs.")
    parser.add_argument("--checkpoint", type=Path, help="Path to HorusEye_demo.pth or compatible checkpoint.")
    parser.add_argument("--window-width", required=True, type=float, help="Display/model window width.")
    parser.add_argument("--window-level", required=True, type=float, help="Display/model window level.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"), help="Inference device.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU model inference. This is slow.")
    parser.add_argument("--smoke-test", action="store_true", help="Skip the model and write a lightweight filtered image.")
    return parser.parse_args()


def first_number(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return float(value[0])
    try:
        return float(value[0])
    except (TypeError, IndexError):
        return float(value)


def read_dicom_slice(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    dataset = pydicom.dcmread(str(path))
    pixel_array = dataset.pixel_array.astype(np.float32)
    if pixel_array.ndim == 3:
        pixel_array = pixel_array[0]

    slope = first_number(getattr(dataset, "RescaleSlope", 1.0), 1.0)
    intercept = first_number(getattr(dataset, "RescaleIntercept", 0.0), 0.0)
    hu = pixel_array * slope + intercept

    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        hu = hu.max() + hu.min() - hu

    pixel_spacing = getattr(dataset, "PixelSpacing", [1.0, 1.0])
    spacing = [first_number(pixel_spacing, 1.0), first_number(pixel_spacing[1], 1.0) if len(pixel_spacing) > 1 else 1.0]

    metadata = {
        "rows": int(hu.shape[0]),
        "columns": int(hu.shape[1]),
        "rescale_slope": float(slope),
        "rescale_intercept": float(intercept),
        "pixel_spacing": spacing,
        "photometric_interpretation": str(getattr(dataset, "PhotometricInterpretation", "")),
        "sop_instance_uid": str(getattr(dataset, "SOPInstanceUID", "")),
    }
    return hu.astype(np.float32, copy=False), metadata


def apply_window(image: np.ndarray, width: float, level: float) -> np.ndarray:
    if width <= 0:
        raise ValueError("Window width must be positive.")
    low = level - width / 2.0
    high = level + width / 2.0
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)


def mean_filter_3x3(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, ((1, 1), (1, 1)), mode="edge")
    total = np.zeros_like(image, dtype=np.float32)
    for row_offset in range(3):
        for col_offset in range(3):
            total += padded[row_offset : row_offset + image.shape[0], col_offset : col_offset + image.shape[1]]
    return total / 9.0


def select_device(requested: str, allow_cpu: bool):
    import torch

    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda:0")
    if requested == "cuda":
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if requested == "cpu" or allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("CUDA is not available. Use --allow-cpu only for small tests.")


def load_state(path: Path, device):
    import torch

    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if isinstance(state, dict) and any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def predict_denoised_slice(windowed_slice: np.ndarray, checkpoint: Path, device) -> np.ndarray:
    import torch
    from monai.inferers import SlidingWindowInferer
    from model import SwinUNet

    if not checkpoint or not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint}")

    model = SwinUNet(in_ch=3)
    model.load_state_dict(load_state(checkpoint, device), strict=True)
    model.eval()
    model.to(device)
    if device.type == "cuda":
        model.half()

    stacked = np.stack((windowed_slice, windowed_slice, windowed_slice), axis=0)[np.newaxis]
    input_set = torch.from_numpy(stacked).to(device=device, dtype=torch.float32)
    if device.type == "cuda":
        input_set = input_set.half()

    with torch.no_grad():
        inferer = SlidingWindowInferer(
            roi_size=(512, 512),
            sw_batch_size=4,
            overlap=0.25,
            mode="gaussian",
            sigma_scale=0.25,
            progress=False,
            sw_device=device,
            device="cpu",
        )
        denoised = inferer(inputs=input_set, network=model).float().detach().cpu().numpy()[0, 0]
    return np.clip(denoised, 0.0, 1.0).astype(np.float32, copy=False)


def save_png(image: np.ndarray, path: Path) -> None:
    display = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(display).save(str(path))


def write_outputs(input_image: np.ndarray, denoised: np.ndarray, output_dir: Path, metadata: dict[str, object]) -> None:
    input_npy = output_dir / "windowed_input.npy"
    denoised_npy = output_dir / "denoised.npy"
    input_png = output_dir / "windowed_input.png"
    denoised_png = output_dir / "denoised.png"
    metadata_path = output_dir / "denoise_outputs.json"

    np.save(input_npy, input_image.astype(np.float32, copy=False))
    np.save(denoised_npy, denoised.astype(np.float32, copy=False))
    save_png(input_image, input_png)
    save_png(denoised, denoised_png)

    metadata.update(
        {
            "outputs": {
                "windowed_input_npy": str(input_npy),
                "denoised_npy": str(denoised_npy),
                "windowed_input_png": str(input_png),
                "denoised_png": str(denoised_png),
            },
            "input_range": [float(input_image.min()), float(input_image.max())],
            "denoised_range": [float(denoised.min()), float(denoised.max())],
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hu, dicom_metadata = read_dicom_slice(args.input_dicom)
    windowed = apply_window(hu, args.window_width, args.window_level)

    metadata = {
        "input_dicom": str(args.input_dicom),
        "window_width": float(args.window_width),
        "window_level": float(args.window_level),
        "smoke_test": bool(args.smoke_test),
        "dicom": dicom_metadata,
    }

    if args.smoke_test:
        denoised = mean_filter_3x3(windowed)
        metadata["device"] = "none"
    else:
        device = select_device(args.device, args.allow_cpu)
        denoised = predict_denoised_slice(windowed, args.checkpoint, device)
        metadata["device"] = str(device)
        metadata["checkpoint"] = str(args.checkpoint)

    write_outputs(windowed, denoised, args.output_dir, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
