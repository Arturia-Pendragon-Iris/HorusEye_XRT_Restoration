import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import cv2


def get_nps(noise_image, show=False):
    f_transform = np.fft.fft2(noise_image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    power_spectrum_2D = np.abs(f_transform_shifted) ** 2 / (10 ** 6)
    phase_spectrum_2D = np.angle(f_transform_shifted)

    power_spectrum_1D = power_spectrum_2D.mean(axis=0)
    phase_spectrum_1D = phase_spectrum_2D.mean(axis=0)
    # print(np.mean(power_spectrum_1D), np.mean(phase_spectrum_1D))

    if show:
        print(power_spectrum_1D[256])

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(power_spectrum_1D)
        plt.title('1D Power-Frequency Spectrum')
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.grid()

        # 绘制一维功率-相位谱
        plt.subplot(1, 2, 2)
        plt.plot(phase_spectrum_1D)
        plt.title('1D Power-Phase Spectrum')
        plt.xlabel('Frequency')
        plt.ylabel('Phase')
        plt.grid()

        plt.tight_layout()
        plt.show()

    return power_spectrum_1D, phase_spectrum_1D


def build_filters():
    filters = []
    ksize = [7, 11, 15, 19]
    lamda = np.pi / 2.0
    for theta in np.arange(0, np.pi, np.pi / 8):
        for K in range(4):
            kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()  # 这里不是很理解
            filters.append(kern)

    return filters


# Gabor特征提取
def getGabor(img):
    filters = build_filters()
    img = img.astype(np.float32)

    res = []  # 滤波结果
    for i in range(len(filters)):
        # res1 = process(img, filters[i])
        accum = np.array(np.zeros_like(img), "float32")
        for kern in filters[i]:
            fimg = cv2.filter2D(img, cv2.CV_32F, kern)
            accum = np.maximum(accum, fimg, accum)

        # res.append(np.asarray(accum))
        res.append(np.mean(accum ** 2))

    return res


def average_list(lst, target_length):
    # Calculate the size of each group
    group_size = len(lst) // target_length

    # Check if the list can be evenly divided
    if len(lst) % target_length != 0:
        raise ValueError("List length must be divisible by the target length.")

    # Calculate the average for each group
    averaged_list = [
        sum(lst[i:i + group_size]) / group_size for i in range(0, len(lst), group_size)
    ]

    return averaged_list
