import torch
import numpy as np
from model import SwinUNet
from monai.inferers import SliceInferer, SlidingWindowInferer


def predict_denoised_slice(ct_slice):
    model = SwinUNet(in_ch=3).cuda()
    model.half()
    model.eval()

    input_set = torch.from_numpy(np.stack((ct_slice, ct_slice, ct_slice), axis=0)[np.newaxis]).to(torch.float)
    input_set = input_set.to('cuda').half()
    model.load_state_dict(torch.load(
        "/data/Model/HorusEye_demo.pth")) # replace the checkpoint path with your own path

    with torch.no_grad():
        inferer = SlidingWindowInferer(roi_size=(512, 512),
                                       sw_batch_size=4,
                                       overlap=0.25,
                                       mode="gaussian",
                                       sigma_scale=0.25,
                                       progress=False,
                                       sw_device="cuda",
                                       device="cpu")
        denoised = inferer(inputs=input_set, network=model).detach().cpu().numpy()[0, 0]
    return np.array(denoised, "float32")
