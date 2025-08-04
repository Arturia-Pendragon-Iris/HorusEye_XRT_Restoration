import torch
import numpy as np


def tv_loss(x):
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.mean(dh[..., :-1, :] + dw[..., :, :-1])


def poisson_loss(y_pred, y_true, photons_per_pixel=4096, mu_max=3071*(20-0.02)/1000+20):
    def get_photons(y):
        y = torch.exp(-y * mu_max) * photons_per_pixel
        return y

    def get_photons_log(y):
        y = -y * mu_max + np.log(photons_per_pixel)
        return y

    y_true_photons = get_photons(y_true)
    y_pred_photons = get_photons(y_pred)
    y_pred_photons_log = get_photons_log(y_pred)
    # print()
    return torch.sum(y_pred_photons - y_true_photons * y_pred_photons_log)
