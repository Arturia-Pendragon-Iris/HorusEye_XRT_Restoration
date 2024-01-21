import argparse, os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import UNet_2D
from filter import do_highpass_filter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda")

def predict_two_slice(slice_1, slice_2, device):

    ct_scan = np.concatenate([slice_1[np.newaxis], slice_2[np.newaxis]], axis=0)
    ct_scan = torch.tensor(ct_scan[np.newaxis, :]).to(torch.float).to(device)
    denoise_model = UNet_2D()
    denoise_model.load_state_dict(torch.load(
        "/data/Model/slice_prediction/spine_epoch_4.pth"))
    denoise_model = denoise_model.cuda()

    prediction = denoise_model(ct_scan).cpu().detach().numpy()[0, 0]

    return prediction


if __name__ == '__main__':
    root_path = "/CT"
    k = 0
    for filename in np.sort(os.listdir(root_path)):
        # if not "L" in filename:
        #     continue
        if k > 150:
            continue
        print(filename)
        raw = np.load(os.path.join(root_path, filename))["arr_0"]
        for i in range(1, raw.shape[-1] - 1):
            predict_i = predict_two_slice(raw[:, :, i - 1], raw[:, :, i + 1], device)
            noise_i = raw[:, :, i] - predict_i

            CT_path = os.path.join("/clean_CT", str(i).zfill(3) + "_" + filename)
            np.savez_compressed(CT_path, predict_i)
            
            denoised_path = os.path.join("/noise", str(i).zfill(3) + "_" + filename)
            np.savez_compressed(denoised_path, noise_i)

        k += 1
