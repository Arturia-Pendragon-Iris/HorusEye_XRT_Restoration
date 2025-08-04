import os.path
from CT_denoise.predict_denoised_v2 import *
from analysis.evaluation import compare_img
import matplotlib.pyplot as plt
from visualization.view_2D import *
from ram import RAM
import torch
import deepinv as dinv
from CT_denoise.common_denoise import *
from HorusEye_R.projection.utils_fb import *
from HorusEye_R.projection.utils import TV_ODL
from CT_denoise.DIPLIB.reconstructors.dip import *
from CT_denoise.DIPLIB.reconstructors.tv import TVReconstructor
import warnings

warnings.filterwarnings("ignore")

device = 'cuda'
model = RAM(device=device).cuda()

root_path = "/mnt/data_1/HorusEye/cleanCT_subset"
psnr_list = []
ssim_list = []
for i in range(50):
    img = np.load(os.path.join(root_path, os.listdir(root_path)[i]))
    # plot_parallel(
    #     a=img
    # )
    # img = np.clip((img + 1000.0) / 1400, 0, 1)
    # img = predict_denoised_slice_test(img)

    # astra_proj_geom, astra_vol_geom, astra_proj_clean = simulate_noisy_proj_astra_fanbeam(img, noise=False)
    # clean_recon = FBP_ASTRA_fanbeam(astra_proj_geom, astra_vol_geom, astra_proj_clean)

    # astra_proj_geom, astra_vol_geom, astra_proj_noisy = simulate_noisy_proj_astra_fanbeam(img, num_angles=720)
    # recon = SART_ASTRA_fanbeam(astra_proj_geom, astra_vol_geom, astra_proj_noisy)
    # recon = predict_denoised_slice_test(recon)
    #
    # proj_max = np.max(astra_proj_noisy)
    # dip = denoise_DIPTV_img(astra_proj_noisy / proj_max) * proj_max
    #
    # plot_parallel(
    #     a=img,
    #     b=recon,
    #     c=img - recon
    # )
    #
    # dip_recon = FBP_ASTRA(astra_proj_geom, astra_vol_geom, dip)
    # plot_parallel(
    #     a=fbp_recon,
    #     b=dip_recon
    # )
    # exit()
    # with open('/home/chuy/PythonProjects/Arturia_platform/CT_denoise/DIPLIB/utils/params/lodopab_tv.json', 'r',
    #           encoding='utf-8') as file:
    #     hyper_params = json.load(file)

    # odl_ray_trafo, odl_proj_noisy = simulate_noisy_proj_odl_fanbeam(img, noise=False, num_angles=720)
    # clean_recon = FBP_ODL(odl_ray_trafo, odl_proj_noisy)
    # #
    # odl_ray_trafo, odl_proj_noisy = simulate_noisy_proj_odl_fanbeam(img)
    # recon = TV_ODL(odl_ray_trafo, odl_proj_noisy, n_iter=200, gamma=1e-4)
    # reconstructor = DIPReconstructor(odl_ray_trafo)
    # recon = reconstructor.reconstruct(odl_proj_noisy)
    # recon = recon * 0.9 + img * 0.1

    # plot_parallel(
    #     a=fbp_recon,
    #     b=img
    # )

    # pnsr, ssim, _, _ = compare_img(recon, img)
    # print(pnsr, ssim)

    astra_proj_geom, astra_vol_geom, astra_proj_noisy = simulate_noisy_proj_astra_fanbeam(img)
    x = FBP_ASTRA_fanbeam(astra_proj_geom, astra_vol_geom, astra_proj_noisy)
    x = torch.from_numpy(np.stack((x, x, x), axis=0)).unsqueeze(0).cuda().to(torch.float32)
    physics = dinv.physics.Demosaicing(img_size=(3, 512, 512),
                                       noise_model=dinv.physics.GaussianNoise(.05), device=device)
    with torch.no_grad():
        recon = model(x, physics=physics)

    x = x.detach().cpu().numpy()[0, 0]
    recon = recon.detach().cpu().numpy()[0, 0]
    #
    psnr, ssim, _, _ = compare_img(img, recon)
    # print(psnr, ssim)

    psnr_list.append(psnr)
    ssim_list.append(ssim)

for i in range(50):
    print(psnr_list[i], ssim_list[i])
