import os.path
from analysis.evaluation import compare_img
import matplotlib.pyplot as plt
from visualization.view_2D import *
from ram import RAM
import torch
import deepinv as dinv
from projection.utils_fb import *
from projection.utils import TV_ODL
from dliplib.reconstructors.dip import *
from dliplib.reconstructors.tv import TVReconstructor
import warnings

warnings.filterwarnings("ignore")

device = 'cuda'
model = RAM(device=device).cuda()

img = np.load("../example_dataset/001.npy", allow_pickle=True)

astra_proj_geom, astra_vol_geom, astra_proj_clean = simulate_noisy_proj_astra_fanbeam(img, noise=True, num_angles=270)
fbp_recon = FBP_ASTRA_fanbeam(astra_proj_geom, astra_vol_geom, astra_proj_clean)
sirt_recon = SIRT_ASTRA_fanbeam(astra_proj_geom, astra_vol_geom, astra_proj_clean, iter=500)
sart_recon = SART_ASTRA_fanbeam(astra_proj_geom, astra_vol_geom, astra_proj_clean, iter=500)

odl_ray_trafo, odl_proj_noisy = simulate_noisy_proj_odl_fanbeam(img)
tv_4_recon = TV_ODL(odl_ray_trafo, odl_proj_noisy, n_iter=15000, gamma=1e-4)
tv_6_recon = TV_ODL(odl_ray_trafo, odl_proj_noisy, n_iter=15000, gamma=2.1e-7)

reconstructor = DIPReconstructor(odl_ray_trafo)
diptv_recon = reconstructor.reconstruct(odl_proj_noisy)

x = torch.from_numpy(np.stack((fbp_recon, fbp_recon, fbp_recon), axis=0)).unsqueeze(0).cuda().to(torch.float32)
physics = dinv.physics.Denoising(img_size=(3, 512, 512),
                                 noise_model=dinv.physics.Tomography(), device=device)
with torch.no_grad():
    recon = model(x, physics=physics)

x = x.detach().cpu().numpy()[0, 0]
ram_recon = recon.detach().cpu().numpy()[0, 0]

plot_parallel(
    a=img,
    b=fbp_recon,
    c=sirt_recon,
    d=sart_recon,
    e=tv_4_recon,
    f=tv_6_recon,
    g=diptv_recon,
    h=ram_recon
)
