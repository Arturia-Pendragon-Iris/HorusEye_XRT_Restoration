import numpy as np
import cupy as cp
import cupyx.scipy.ndimage
import torch
import torch.nn.functional as F


def zoom_with_cupy(input_array, zoom_factor, order=2):
    cupy_array = cp.array(input_array)
    zoomed_cupy_array = cupyx.scipy.ndimage.zoom(cupy_array, zoom_factor, order=order)
    zoomed_numpy_array = cp.asnumpy(zoomed_cupy_array)

    return zoomed_numpy_array


def zoom_with_torch(input_array, zoom_factor):
    tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to("cuda")
    # 使用 F.interpolate 进行缩放
    scaled_tensor = F.interpolate(tensor, scale_factor=zoom_factor, mode='trilinear', align_corners=False)

    # 将结果移动回 CPU 并转换回 NumPy 数组
    scaled_array = scaled_tensor[0, 0].cpu().numpy()
    return scaled_array


