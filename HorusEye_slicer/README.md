# DICOM Denoise Demo

This folder contains a minimal 3D Slicer demo for denoising a single DICOM CT image.

The demo takes:

- one DICOM file,
- a window width,
- a window level,
- an external Python environment,
- and an optional denoising checkpoint.

It writes:

- `windowed_input.npy`
- `denoised.npy`
- `windowed_input.png`
- `denoised.png`
- `denoise_outputs.json`

The Slicer module then loads `denoised.npy` back into the scene as a single-slice scalar volume.

## Setup

Add this folder to Slicer's additional module paths:

```text
<repo-root>\DICOMDenoiseDemo
```

Restart Slicer and open `DICOM Denoise Demo`.

The external Python environment should contain:

```text
torch
monai
pydicom
Pillow
numpy
```

Example:

```powershell
conda activate arturia_v1
pip install monai pydicom Pillow numpy
```

Install the PyTorch build that matches your CUDA runtime.

## Checkpoint

The original inference snippet expects:

```text
/data/Model/HorusEye_demo.pth
```

For this Slicer demo, the checkpoint path is configurable in the UI. Select `HorusEye_demo.pth` or another checkpoint compatible with `SwinUNet(in_ch=3)`.

## Smoke Test

Smoke test mode skips the neural network and applies a tiny 3x3 mean filter. It is only intended to verify DICOM reading, windowing, output writing, and Slicer loading.

Command-line example:

```powershell
& %USERPROFILE%\miniconda3\envs\arturia_v1\python.exe `
  <repo-root>\DICOMDenoiseDemo\denoise_inference.py `
  --input-dicom path\to\slice.dcm `
  --output-dir <repo-root>\denoise_smoke_outputs `
  --window-width 400 `
  --window-level 40 `
  --smoke-test
```

Smoke-test output is not a real denoising result.
