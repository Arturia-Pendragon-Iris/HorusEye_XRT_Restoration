import torch
import numpy as np
from model import SwinUNet
from monai.inferers import SliceInferer, SlidingWindowInferer


def predict_denoised_slice(ct_slice):
    model = SwinUNet().cuda()
    model.load_state_dict(torch.load("/data/Model/denoise_V10/HorusEye.pth"))
    model.half()
    model.eval()

    input_set = torch.tensor(np.stack([ct_slice] * 3, axis=0)[np.newaxis, :]).to(torch.float).to('cuda').half()
    with torch.no_grad():
        inferer = SlidingWindowInferer(roi_size=(512, 512),
                                       sw_batch_size=1,
                                       overlap=0.25,
                                       mode="gaussian",
                                       sigma_scale=0.25,
                                       progress=False,
                                       sw_device="cuda",
                                       device="cpu")

        denoised = inferer(inputs=input_set, network=model).detach().cpu().numpy()[0, 0]
    return np.array(np.mean(denoised, axis=0), "float32")


def predict_denoised_volume(ct_array):
    model = SwinUNet().cuda()
    model.half()
    model.eval()

    input_set = torch.from_numpy(ct_array[np.newaxis, np.newaxis]).to(torch.float)
    input_set = input_set.to('cuda').half()
    model.load_state_dict(torch.load("/data/Model/denoise_V10/HorusEye.pth"))
    with torch.no_grad():
        inferer = SliceInferer(spatial_dim=2,
                               roi_size=(512, 512),
                               sw_batch_size=4,
                               progress=False)
        denoised = inferer(inputs=input_set, network=model).detach().cpu().numpy()[0, 0]

    return np.array(denoised, "float32")
