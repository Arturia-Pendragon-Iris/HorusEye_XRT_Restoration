from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_bilateral,
    denoise_wavelet,
    denoise_nl_means,
    estimate_sigma,
)
import pywt
from sklearn.feature_extraction import image
from ksvd import ApproximateKSVD
from analysis.filter.filter_2D import gaussian_filter
import cv2
from bm3d import bm3d, BM3DProfile
import numpy as np
from sklearn import linear_model
from scipy.ndimage import gaussian_filter, sobel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import tqdm
from DIP.utils.common_utils import get_noise, get_image, np_to_torch
from DIP.models.skip import skip
import prox_tv as ptv

cv2.setUseOptimized(True)


def denoise_tv_img(img, weight=0.01):
    de = denoise_tv_chambolle(img[np.newaxis], weight=weight, channel_axis=0)
    return de[0]


def denoise_tv_scan(img, weight=0.07):
    de = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    for i in range(img.shape[-1]):
        de[:, :, i] = denoise_tv_img(img[:, :, i], weight=weight)
    return np.array(de, "float32")


def denoise_atv_img(image, alpha0=0.05, weight_range=(1e-4, 0.1), tol=1e-6, max_iter=10):
    """
    TV denoising with automatic weight selection using residual-based noise estimation.

    Parameters:
        image: 2D ndarray, float32/float64, normalized to [0, 1]
        alpha0: Initial small weight for first-pass denoising
        weight_range: Search range for TV weight (tuple)
        tol: Tolerance for MSE matching
        max_iter: Max iterations for binary search

    Returns:
        Denoised image (2D ndarray)
    """
    # Step 1: Estimate noise variance using small weight
    u0 = denoise_tv_chambolle(image, weight=alpha0)
    sigma2 = np.mean((image - u0) ** 2)

    # Step 2: Binary search for weight matching residual ~ sigma²
    lo, hi = weight_range
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        u = denoise_tv_chambolle(image, weight=mid)
        mse = np.mean((image - u) ** 2)
        print(mse, sigma2)
        if abs(mse - sigma2) < tol:
            break
        if mse > sigma2:
            lo = mid
        else:
            hi = mid

    return u


def denoise_bila_img(img):
    de = denoise_bilateral(img[np.newaxis], sigma_color=0.05, sigma_spatial=15, channel_axis=0)
    return de[0]


def denoise_wave_img(img):
    de = denoise_wavelet(img[np.newaxis], channel_axis=0, rescale_sigma=True)
    return de[0]


def denoise_wave_scan(img):
    de = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    for i in range(img.shape[-1]):
        de[:, :, i] = denoise_wave(img[:, :, i])
    return np.array(de, "float32")


def denoise_nl_img(img):
    de = denoise_nl_means(img)
    return de


def denoise_nl_scan(img):
    de = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    for i in range(img.shape[-1]):
        de[:, :, i] = denoise_nl_img(img[:, :, i])
    return np.array(de, "float32")


def denoise_gaussian_img(img, sigma=0.5):
    return gaussian_filter(img, sigma=sigma)


def denoise_gaussian_scan(img):
    de = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    for i in range(img.shape[-1]):
        de[:, :, i] = denoise_gaussian_img(img[:, :, i])
    return np.array(de, "float32")


##### KSVD denoising
class KSVD(object):
    def __init__(self, n_components, max_iter=30, tol=5000,
                 n_nonzero_coefs=None):
        self.dictionary = None
        self.sparsecode = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs

    def _initialize(self, y):
        u, s, v = np.linalg.svd(y)
        self.dictionary = u[:, :self.n_components]
        # print(self.dictionary.shape)

    def _update_dict(self, y, d, x):
        for i in range(self.n_components):
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue

            d[:, i] = 0
            r = (y - np.dot(d, x))[:, index]
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0].T
            x[i, index] = s[0] * v[0, :]
        return d, x

    def fit(self, y):
        """
        KSVD迭代过程
        """
        self._initialize(y)
        for i in range(self.max_iter):
            x = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
            e = np.linalg.norm(y - np.dot(self.dictionary, x))
            if e < self.tol:
                break
            self._update_dict(y, self.dictionary, x)

        self.sparsecode = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
        return self.dictionary, self.sparsecode


def denoise_ksvd_img(img):
    patch_size = (5, 5)
    patches = image.extract_patches_2d(img, patch_size)
    signals = patches.reshape(patches.shape[0], -1)
    mean = np.mean(signals, axis=1)[:, np.newaxis]
    signals -= mean
    aksvd = ApproximateKSVD(n_components=32)
    dictionary = aksvd.fit(signals[:1000]).components_
    gamma = aksvd.transform(signals)
    reduced = gamma.dot(dictionary) + mean
    reduced_img = image.reconstruct_from_patches_2d(
        reduced.reshape(patches.shape), img.shape)

    return reduced_img


def denoise_ksvd_2_img(img, patch_size=(5, 5), n_components=32, n_nonzero=5):
    patches = image.extract_patches_2d(img, patch_size)
    signals = patches.reshape(patches.shape[0], -1).astype(np.float32)
    mean = np.mean(signals, axis=1)[:, np.newaxis]
    signals -= mean

    aksvd = ApproximateKSVD(n_components=n_components, transform_n_nonzero_coefs=n_nonzero)
    subset = signals[np.random.choice(len(signals), min(1000, len(signals)), replace=False)]
    dictionary = aksvd.fit(subset).components_

    gamma = aksvd.transform(signals)
    reduced = gamma.dot(dictionary) + mean
    reduced_img = image.reconstruct_from_patches_2d(reduced.reshape(patches.shape), img.shape)

    return reduced_img


def denoise_ksvd_scan(img):
    de = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    for i in range(img.shape[-1]):
        de[:, :, i] = denoise_ksvd_img(img[:, :, i])
    return np.array(de, "float32")


class BM3D:
    def __init__(self, sigma=25):
        self.sigma = sigma
        self.Thresh_Hard3D = 2.7 * sigma
        self.Step1_thresh = 2500
        self.Step1_max_cnt = 16
        self.Step1_blk_size = 8
        self.Step1_blk_step = 3
        self.Step1_search_step = 3
        self.Step1_window = 39
        self.Step2_thresh = 400
        self.Step2_max_cnt = 32
        self.Step2_blk_size = 8
        self.Step2_blk_step = 3
        self.Step2_search_step = 3
        self.Step2_window = 39
        self.beta_kaiser = 2.0

    def init_arrays(self, img, blk_size):
        shape = img.shape
        accum = np.zeros(shape, dtype=float)
        weight = np.zeros(shape, dtype=float)
        kaiser = np.outer(np.kaiser(blk_size, self.beta_kaiser), np.kaiser(blk_size, self.beta_kaiser))
        return accum, weight, kaiser

    def locate_block(self, i, j, step, blk_size, width, height):
        x = i * step if i * step + blk_size < width else width - blk_size
        y = j * step if j * step + blk_size < height else height - blk_size
        return np.array((x, y), dtype=int)

    def search_window(self, img, point, window_size, blk_size):
        x, y = point
        lx = max(x + blk_size // 2 - window_size // 2, 0)
        ly = max(y + blk_size // 2 - window_size // 2, 0)
        lx = min(lx, img.shape[0] - window_size)
        ly = min(ly, img.shape[1] - window_size)
        return np.array((lx, ly), dtype=int)

    def dct_match(self, img, ref_blk, ref_pos, blk_size, step, threshold, max_cnt, window_size):
        ref_dct = cv2.dct(ref_blk.astype(np.float64))
        matched = np.zeros((max_cnt, blk_size, blk_size), dtype=float)
        positions = np.zeros((max_cnt, 2), dtype=int)
        matched[0] = ref_dct
        positions[0] = ref_pos

        win_loc = self.search_window(img, ref_pos, window_size, blk_size)
        bx, by = win_loc
        blk_num = (window_size - blk_size) // step
        sim_blks = np.zeros((blk_num ** 2, blk_size, blk_size), dtype=float)
        sim_pos = np.zeros((blk_num ** 2, 2), dtype=int)
        dists = np.zeros(blk_num ** 2, dtype=float)
        cnt = 0

        for i in range(blk_num):
            for j in range(blk_num):
                x, y = bx + i * step, by + j * step
                blk = cv2.dct(img[x:x + blk_size, y:y + blk_size].astype(np.float64))
                dist = np.linalg.norm(ref_dct - blk) ** 2 / blk_size ** 2
                if 0 < dist < threshold:
                    sim_blks[cnt], sim_pos[cnt], dists[cnt] = blk, (x, y), dist
                    cnt += 1

        top = min(cnt + 1, max_cnt)
        sorted_idx = np.argsort(dists[:cnt])
        for i in range(1, top):
            matched[i] = sim_blks[sorted_idx[i - 1]]
            positions[i] = sim_pos[sorted_idx[i - 1]]
        return matched, positions, top

    def thresholding(self, blocks):
        nonzero = 0
        for i in range(blocks.shape[1]):
            for j in range(blocks.shape[2]):
                coeff = cv2.dct(blocks[:, i, j])
                coeff[np.abs(coeff) < self.Thresh_Hard3D] = 0.
                nonzero += np.count_nonzero(coeff)
                blocks[:, i, j] = cv2.idct(coeff)[0]
        return blocks, nonzero

    def aggregate(self, blocks, positions, accum, weight, nonzero, count, kaiser):
        if nonzero < 1:
            nonzero = 1
        blk_weight = (1. / nonzero) * kaiser
        for i in range(count):
            x, y = positions[i]
            blk = (1. / nonzero) * cv2.idct(blocks[i]) * kaiser
            accum[x:x + blk.shape[0], y:y + blk.shape[1]] += blk
            weight[x:x + blk.shape[0], y:y + blk.shape[1]] += blk_weight

    def step1(self, noisy_img):
        h, w = noisy_img.shape
        step = self.Step1_blk_step
        blk_size = self.Step1_blk_size
        blk_count_x = (h - blk_size) // step + 2
        blk_count_y = (w - blk_size) // step + 2
        accum, weight, kaiser = self.init_arrays(noisy_img, blk_size)

        for i in range(blk_count_x):
            for j in range(blk_count_y):
                pt = self.locate_block(i, j, step, blk_size, h, w)
                blk = noisy_img[pt[0]:pt[0] + blk_size, pt[1]:pt[1] + blk_size]
                matched, pos, count = self.dct_match(noisy_img, blk, pt, blk_size,
                                                     self.Step1_search_step, self.Step1_thresh,
                                                     self.Step1_max_cnt, self.Step1_window)
                matched, nonzero = self.thresholding(matched)
                self.aggregate(matched, pos, accum, weight, nonzero, count, kaiser)

        return np.uint8(np.divide(accum, weight, out=np.zeros_like(accum), where=weight != 0))


# def denoise_bm3d_img(img, var=0.1):
#     denoised_1 = bm3d.bm3d(img, sigma_psd=var)
#     e_var = np.std(denoised_1 - img)
#     return bm3d.bm3d(img, sigma_psd=e_var)


# def denoise_bm3d_scan(img):
#     de = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
#     for i in range(img.shape[-1]):
#         de[:, :, i] = denoise_bm3d_img(img[:, :, i])
#     return np.array(de, "float32")


def denoise_bm3d_img(img):
    def estimate_sigma_mad(image):
        coeffs = pywt.dwt2(image, 'db1')
        _, (cH, cV, cD) = coeffs
        high_freq = np.concatenate([cH.ravel(), cV.ravel(), cD.ravel()])
        sigma_est = np.median(np.abs(high_freq)) / 0.6745
        return sigma_est

    print(estimate_sigma_mad(img))
    return bm3d(img, sigma_psd=estimate_sigma_mad(img))


def denoise_bm3d_img_2(img, var=0.05):
    return bm3d(img, sigma_psd=var)


class ZS_N2N(nn.Module):
    def __init__(self, n_channels=1, embed_channels=48):
        super(ZS_N2N, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_channels, embed_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(embed_channels, embed_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(embed_channels, n_channels, kernel_size=1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)

    def _pair_downsampler(self, img):
        # Input shape: (B, C, H, W)
        c = img.shape[1]
        f1 = torch.tensor([[[[0, 0.5], [0.5, 0]]]], dtype=img.dtype, device=img.device).repeat(c, 1, 1, 1)
        f2 = torch.tensor([[[[0.5, 0], [0, 0.5]]]], dtype=img.dtype, device=img.device).repeat(c, 1, 1, 1)
        out1 = F.conv2d(img, f1, stride=2, groups=c)
        out2 = F.conv2d(img, f2, stride=2, groups=c)
        return out1, out2

    def _mse(self, x, y):
        return F.mse_loss(x, y)

    def compute_loss(self, noisy_img):
        # Residual prediction loss
        noisy1, noisy2 = self._pair_downsampler(noisy_img)
        pred1 = noisy1 - self(noisy1)
        pred2 = noisy2 - self(noisy2)
        loss_res = 0.5 * (self._mse(noisy1, pred2) + self._mse(noisy2, pred1))

        # Consistency loss
        denoised = noisy_img - self(noisy_img)
        denoised1, denoised2 = self._pair_downsampler(denoised)
        loss_cons = 0.5 * (self._mse(pred1, denoised1) + self._mse(pred2, denoised2))

        return loss_res + loss_cons

    def train_step(self, optimizer, noisy_img):
        self.train()
        loss = self.compute_loss(noisy_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


def denoise_ZSN2N_img(img):
    # Normalize and prepare input
    img = np.clip(img, 0, 1)
    img = torch.tensor(img[np.newaxis, np.newaxis], dtype=torch.float32).cuda()

    # Initialize model
    model = ZS_N2N(n_channels=1).cuda()

    # Training settings
    max_epoch = 1000
    lr = 0.001
    step_size = 1000
    gamma = 0.5

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training loop
    for epoch in range(max_epoch):
        model.train_step(optimizer, img)
        scheduler.step()

    # Inference
    model.eval()
    with torch.no_grad():
        denoised = img - model(img)

    return denoised.cpu().numpy()[0, 0]


def denoise_ZSN2N_scan(img):
    de = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    for i in range(img.shape[-1]):
        de[:, :, i] = denoise_ZSN2N_img(img[:, :, i])
    return np.array(de, "float32")


def denoise_DIP_img(img, num_iter=1200, lr=0.01, input_depth=32):
    # input_np: 2D array in [0,1]
    img_np = np.array(img, "float32")
    img_torch = np_to_torch(img_np)[None].cuda()

    net = skip(input_depth, img_torch.shape[1],
               num_channels_down=[128, 128, 128, 128],
               num_channels_up=[128, 128, 128, 128],
               num_channels_skip=[4, 4, 4, 4],
               upsample_mode='bilinear').cuda()
    # net.apply(lambda m: m.weight.data.normal_())

    noise = get_noise(input_depth, 'noise', img_np.shape).cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for i in range(num_iter):
        optimizer.zero_grad()
        out = net(noise)
        loss = torch.nn.functional.mse_loss(out, img_torch)
        loss.backward()
        optimizer.step()
        # if i % 500 == 0:
        #     print(f"[{i}/{num_iter}] loss: {loss.item():.6f}")

    out_np = out.detach().cpu().numpy()[0, 0]
    return np.clip(out_np, 0, 1)


def denoise_DIP_scan(volume, num_iter=600, lr=0.02, input_depth=32):
    de = np.zeros([volume.shape[0], volume.shape[1], volume.shape[2]])
    for i in range(volume.shape[-1]):
        de[:, :, i] = denoise_DIP_img(volume[:, :, i], num_iter, lr, input_depth)
    return np.array(de, "float32")

