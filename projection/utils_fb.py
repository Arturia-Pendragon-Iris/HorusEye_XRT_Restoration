import odl
import astra
import numpy as np


def simulate_noisy_proj_odl_fanbeam(img, I0=1e5, num_angles=360, detector_count=984,
                                    DSO=1400.0, DOD=500.0,
                                    noise=True):
    image_size = img.shape[0]

    odl_reco_space = odl.uniform_discr(
        [-1, -1], [1, 1], [image_size, image_size], dtype='float32'
    )
    odl_phantom = odl_reco_space.element(img)

    odl_angle_partition = odl.uniform_partition(0.0, np.pi, num_angles)

    detector_length = 4.0
    odl_detector_partition = odl.uniform_partition(-detector_length / 2, detector_length / 2, detector_count)

    odl_geometry = odl.tomo.FanBeamGeometry(
        odl_angle_partition, odl_detector_partition,
        src_radius=DSO, det_radius=DOD
    )

    odl_ray_trafo = odl.tomo.RayTransform(
        odl_reco_space, odl_geometry, impl='astra_cuda'
    )

    odl_proj_data = odl_ray_trafo(odl_phantom)
    if not noise:
        return odl_ray_trafo, odl_proj_data

    odl_intensity = I0 * np.exp(-odl_proj_data)
    odl_noisy_intensity = np.random.poisson(odl_intensity)
    odl_noisy_intensity[odl_noisy_intensity == 0] = 1  # 避免 log(0)
    odl_proj_noisy = -np.log(odl_noisy_intensity / I0)
    odl_proj_space = odl_ray_trafo.range
    odl_proj_noisy = odl_proj_space.element(odl_proj_noisy)

    return odl_ray_trafo, odl_proj_noisy


def simulate_noisy_proj_astra_fanbeam(img, I0=1e5, num_angles=720, detector_count=984,
                                      DSO=1400.0, DOD=500.0,
                                      noise=True):
    image_size = img.shape[0]

    astra_vol_geom = astra.create_vol_geom(image_size, image_size)
    astra_angles = np.linspace(-np.pi, np.pi, num_angles, endpoint=False)

    det_width = 1.0

    astra_proj_geom = astra.create_proj_geom(
        'fanflat',
        det_width, detector_count, astra_angles,
        DSO, DOD
    )

    astra_image_id = astra.data2d.create('-vol', astra_vol_geom, data=img)

    astra_sinogram_id = astra.data2d.create('-sino', astra_proj_geom)
    astra_proj_id = astra.create_projector('line_fanflat', astra_proj_geom, astra_vol_geom)

    cfg = astra.astra_dict('FP')
    cfg['ProjectorId'] = astra_proj_id
    cfg['VolumeDataId'] = astra_image_id
    cfg['ProjectionDataId'] = astra_sinogram_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    astra_proj_data = astra.data2d.get(astra_sinogram_id)

    if not noise:
        return astra_proj_geom, astra_vol_geom, astra_proj_data

    scale_ratio = 0.00390898
    astra_proj_data = astra_proj_data * scale_ratio

    astra_intensity = I0 * np.exp(-astra_proj_data)
    astra_noisy_intensity = np.random.poisson(astra_intensity)
    astra_noisy_intensity[astra_noisy_intensity == 0] = 1
    astra_proj_noisy = -np.log(astra_noisy_intensity / I0) / scale_ratio

    return astra_proj_geom, astra_vol_geom, astra_proj_noisy


def FBP_ASTRA_fanbeam(astra_proj_geom, astra_vol_geom, astra_proj_noisy):
    sinogram_id = astra.data2d.create('-sino', astra_proj_geom, data=astra_proj_noisy)
    recon_id = astra.data2d.create('-vol', astra_vol_geom)
    cfg_fbp = astra.astra_dict('FBP_CUDA')

    cfg_fbp['ReconstructionDataId'] = recon_id
    cfg_fbp['ProjectionDataId'] = sinogram_id
    cfg_fbp['ProjectorId'] = astra.create_projector('line_fanflat', astra_proj_geom, astra_vol_geom)

    alg_id_fbp = astra.algorithm.create(cfg_fbp)
    astra.algorithm.run(alg_id_fbp)
    fbp_recon = astra.data2d.get(recon_id)

    astra.algorithm.delete(alg_id_fbp)
    astra.data2d.delete(recon_id)

    return fbp_recon


def SIRT_ASTRA_fanbeam(astra_proj_geom, astra_vol_geom, astra_proj_noisy, iter=200):
    recon_id = astra.data2d.create('-vol', astra_vol_geom)
    sinogram_id = astra.data2d.create('-sino', astra_proj_geom, data=astra_proj_noisy)

    cfg_sirt = astra.astra_dict('SIRT_CUDA' if astra.astra.use_cuda() else 'SIRT')
    cfg_sirt['ReconstructionDataId'] = recon_id
    cfg_sirt['ProjectionDataId'] = sinogram_id
    cfg_sirt['ProjectorId'] = astra.create_projector('line_fanflat', astra_proj_geom, astra_vol_geom)

    alg_id_sirt = astra.algorithm.create(cfg_sirt)
    astra.algorithm.run(alg_id_sirt, iter)

    sirt_recon = astra.data2d.get(recon_id)
    astra.algorithm.delete(alg_id_sirt)
    astra.data2d.delete(recon_id)
    return sirt_recon


def SART_ASTRA_fanbeam(astra_proj_geom, astra_vol_geom, astra_proj_noisy, iter=200):
    recon_id = astra.data2d.create('-vol', astra_vol_geom)
    sinogram_id = astra.data2d.create('-sino', astra_proj_geom, data=astra_proj_noisy)

    cfg_sart = astra.astra_dict('SART_CUDA' if astra.astra.use_cuda() else 'SART')
    cfg_sart['ReconstructionDataId'] = recon_id
    cfg_sart['ProjectionDataId'] = sinogram_id
    cfg_sart['ProjectorId'] = astra.create_projector('line_fanflat', astra_proj_geom, astra_vol_geom)

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
