import numpy as np
import odl
import astra
from analysis.evaluation import compare_img
import matplotlib.pyplot as plt
from visualization.view_2D import *
import json
from tqdm import tqdm
import warnings
from dliplib.reconstructors.tv import TVReconstructor

warnings.filterwarnings("ignore")


def simulate_noisy_proj_odl(img, I0=1e5, num_angles=360, detector_count=984, noise=True):
    image_size = img.shape[0]
    odl_reco_space = odl.uniform_discr(
        [-1, -1], [1, 1], [image_size, image_size], dtype='float32'
    )
    odl_phantom = odl_reco_space.element(img)

    odl_angle_partition = odl.uniform_partition(0.0, np.pi, num_angles)
    odl_detector_partition = odl.uniform_partition(-2, 2, detector_count)
    odl_geometry = odl.tomo.Parallel2dGeometry(
        odl_angle_partition, odl_detector_partition
    )

    odl_ray_trafo = odl.tomo.RayTransform(
        odl_reco_space, odl_geometry, impl='astra_cpu'
    )

    odl_proj_data = odl_ray_trafo(odl_phantom)
    if not noise:
        return odl_ray_trafo, odl_proj_data

    odl_intensity = I0 * np.exp(-odl_proj_data)
    odl_noisy_intensity = np.random.poisson(odl_intensity)
    odl_noisy_intensity[odl_noisy_intensity == 0] = 1  # 防止 log(0)
    odl_proj_noisy = -np.log(odl_noisy_intensity / I0)

    odl_proj_space = odl_ray_trafo.range
    odl_proj_noisy = odl_proj_space.element(odl_proj_noisy)
    return odl_ray_trafo, odl_proj_noisy


def simulate_noisy_proj_astra(img, I0=1e5, num_angles=360, detector_count=984, noise=True):
    image_size = img.shape[0]
    astra_vol_geom = astra.create_vol_geom(image_size, image_size)
    astra_angles = np.linspace(-np.pi, np.pi, num_angles, endpoint=False)
    astra_proj_geom = astra.create_proj_geom('parallel', 1.0, detector_count, astra_angles)
    astra_image_id = astra.data2d.create('-vol', astra_vol_geom, data=img)
    astra_sinogram_id = astra.data2d.create('-sino', astra_proj_geom)
    astra_proj_id = astra.create_projector('linear', astra_proj_geom, astra_vol_geom)

    cfg = astra.astra_dict('FP')
    cfg['ProjectorId'] = astra_proj_id
    cfg['VolumeDataId'] = astra_image_id
    cfg['ProjectionDataId'] = astra_sinogram_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    astra_proj_data = astra.data2d.get(astra_sinogram_id)
    if not noise:
        return astra_proj_geom, astra_vol_geom, astra_proj_data
    # scale_ratio = np.max(odl_proj_data) / np.max(astra_proj_data)
    scale_ratio = 0.00390898 # normalization scale between astra and odl
    astra_proj_data = astra_proj_data * scale_ratio

    astra_intensity = I0 * np.exp(-astra_proj_data)
    astra_noisy_intensity = np.random.poisson(astra_intensity)
    astra_noisy_intensity[astra_noisy_intensity == 0] = 1
    astra_proj_noisy = -np.log(astra_noisy_intensity / I0) / scale_ratio

    return astra_proj_geom, astra_vol_geom, astra_proj_noisy


def FBP_ASTRA(astra_proj_geom, astra_vol_geom, astra_proj_noisy):
    sinogram_id = astra.data2d.create('-sino', astra_proj_geom, data=astra_proj_noisy)
    recon_id = astra.data2d.create('-vol', astra_vol_geom)

    cfg_fbp = astra.astra_dict('FBP_CUDA')
    cfg_fbp['ReconstructionDataId'] = recon_id
    cfg_fbp['ProjectionDataId'] = sinogram_id
    cfg_fbp['ProjectorId'] = astra.create_projector('linear', astra_proj_geom, astra_vol_geom)
    alg_id_fbp = astra.algorithm.create(cfg_fbp)
    astra.algorithm.run(alg_id_fbp)
    fbp_recon = astra.data2d.get(recon_id)

    astra.algorithm.delete(alg_id_fbp)
    astra.data2d.delete(recon_id)

    return fbp_recon


def SIRT_ASTRA(astra_proj_geom, astra_vol_geom, astra_proj_noisy, iter=200):
    recon_id = astra.data2d.create('-vol', astra_vol_geom)
    sinogram_id = astra.data2d.create('-sino', astra_proj_geom, data=astra_proj_noisy)

    cfg_sirt = astra.astra_dict('SIRT_CUDA' if astra.astra.use_cuda() else 'SIRT')
    cfg_sirt['ReconstructionDataId'] = recon_id
    cfg_sirt['ProjectionDataId'] = sinogram_id
    cfg_sirt['ProjectorId'] = astra.create_projector('linear', astra_proj_geom, astra_vol_geom)

    alg_id_sirt = astra.algorithm.create(cfg_sirt)
    astra.algorithm.run(alg_id_sirt, iter)

    sirt_recon = astra.data2d.get(recon_id)
    astra.algorithm.delete(alg_id_sirt)
    astra.data2d.delete(recon_id)
    return sirt_recon


def SART_ASTRA(astra_proj_geom, astra_vol_geom, astra_proj_noisy, iter=200):
    recon_id = astra.data2d.create('-vol', astra_vol_geom)
    sinogram_id = astra.data2d.create('-sino', astra_proj_geom, data=astra_proj_noisy)

    cfg_sart = astra.astra_dict('SART_CUDA' if astra.astra.use_cuda() else 'SART')
    cfg_sart['ReconstructionDataId'] = recon_id
    cfg_sart['ProjectionDataId'] = sinogram_id
    cfg_sart['ProjectorId'] = astra.create_projector('linear', astra_proj_geom, astra_vol_geom)

    alg_id_sart = astra.algorithm.create(cfg_sart)
    astra.algorithm.run(alg_id_sart, iter)
    sart_recon = astra.data2d.get(recon_id)
    astra.algorithm.delete(alg_id_sart)
    astra.data2d.delete(recon_id)

    return sart_recon


def FBP_ODL(ray_trafo, proj_data):
    pseudoinverse = odl.tomo.fbp_op(ray_trafo)
    fbp_recon = pseudoinverse(proj_data)

    return fbp_recon


def TV_ODL(ray_trafo, noisy_proj, gamma=1e-4, n_iter=200):
    proj_space = ray_trafo.range
    noisy_proj = proj_space.element(noisy_proj)

    tv_reconstructor = TVReconstructor(ray_trafo, gamma=gamma, iterations=n_iter)

    tv_recon = tv_reconstructor.reconstruct(noisy_proj)

    return np.array(tv_recon)


def TV_J(real):
    import numpy as np
    from skimage.restoration import (calibrate_denoiser,
                                     denoise_wavelet)
    from functools import partial

    _denoise_wavelet = partial(denoise_wavelet, rescale_sigma=True)
    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {'sigma': np.linspace(0.001, 0.0000001, 15),
                        'wavelet': ['haar', 'sym2'],
                        'mode': ['soft'],
                        'wavelet_levels': [2],
                        'method': ['BayesShrink', 'VisuShrink']}

    calibrated_denoiser = calibrate_denoiser(real,
                                             _denoise_wavelet,
                                             denoise_parameters=parameter_ranges
                                             )

    calibrated_output = calibrated_denoiser(real)
    return calibrated_output


def SGM(img, ray_trafo, proj_data):
    import torch
    import os
    from CT_denoise.SGM.models.cond_refinenet_dilated_noconv import CondRefineNetDilated
    import torch.nn as nn

    states = torch.load(os.path.join('/data/Model/denoise_V9/SGM.pth'), map_location="cuda")
    scorenet = CondRefineNetDilated().cuda()
    model_dict = scorenet.state_dict()
    # Delete keys in pretrained_dict that are not model_dict
    pretrained_dict = {k: v for k, v in states[0].items() if k in model_dict}
    model_dict.update(pretrained_dict)

    scorenet.load_state_dict(model_dict)
    scorenet.eval()

    image_shape = list((1,) + (10,) + img.shape[0:2])
    x0 = nn.Parameter(torch.Tensor(np.zeros(image_shape)).uniform_(-1, 1)).cuda()
    x01 = x0

    x = np.copy(img)
    z = np.copy(x)

    sigmas = np.exp(np.linspace(np.log(1), np.log(0.05), 6))
    n_steps = 25
    step_lr = 0.00005

    maxdegrade = np.max(img)
    ATA = ray_trafo.adjoint(ray_trafo(ray_trafo.domain.one()))

    with torch.no_grad():
        for idx, sigma in enumerate(sigmas):
            # print(idx)
            lambda_recon = 1. / sigma ** 2
            labels = torch.ones(1, device=x0.device) * idx
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            # print('sigma = {}'.format(sigma))
            ### SQS
            for step in range(n_steps):
                noise1 = torch.rand_like(x0) * np.sqrt(step_size * 2)
                grad1 = scorenet(x01, labels).detach()
                x0 = x0 + step_size * grad1
                x01 = x0 + noise1
                # print(step_size)
                x0 = np.array(x0.cpu().detach(), dtype=np.float32)
                x1 = np.squeeze(x0)
                x1 = np.mean(x1, axis=0)

                hyper = 350
                sum_diff = x - x1 * maxdegrade

                norm_diff = ray_trafo.adjoint((ray_trafo(x) - proj_data))
                x_new = z - (norm_diff + 2 * hyper * sum_diff) / (ATA + 2 * hyper)
                z = x_new + 0.5 * (x_new - x)
                x = x_new
                x_rec = x.asarray()
                x_rec = x_rec / maxdegrade

                if (step % 10) == 0:
                    x_rec1 = TV_J(x_rec)
                    x_rec = x_rec + sigma * (x_rec1 - x_rec)

                x_mid = np.zeros([1, 10, 512, 512], dtype=np.float32)
                x_rec = np.clip(x_rec, 0, 1)
                x_rec = np.expand_dims(x_rec, 2)
                x_mid_1 = np.tile(x_rec, [1, 1, 10])
                x_mid_1 = np.transpose(x_mid_1, [2, 0, 1])
                x_mid[0, :, :, :] = x_mid_1
                x0 = torch.tensor(x_mid, dtype=torch.float32).cuda()
            out = x_rec

    return out[:, :, 0]
