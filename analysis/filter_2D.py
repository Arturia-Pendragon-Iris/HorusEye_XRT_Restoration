import numpy as np
from skimage.filters import frangi
from scipy import ndimage
# from analysis.get_surface import get_surface_3D
from scipy.ndimage import gaussian_filter as gaussian
from scipy.ndimage import sobel
import cv2
from visualization.view_2D import plot_parallel


# Gaussian filter
def gaussian_filter(np_array, sigma=2, order=0):
    return gaussian(np_array, sigma=sigma, order=order)


def sobel_filter(np_array):
    sobel_h = ndimage.sobel(np_array, 0)  # horizontal gradient
    sobel_v = ndimage.sobel(np_array, 1)  # vertical gradient
    magnitude = np.sqrt(sobel_h ** 2 + sobel_v ** 2)

    return magnitude / np.max(magnitude)


# kernel filter
def do_highpass_filter(np_array, order=3, average=True):
    # cv2.imshow("org", image)
    if order == 3:
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
    elif order == 5:
        kernel = np.array([[-1, -1, -1, -1, -1],
                           [-1, 1, 2, 1, -1],
                           [-1, 2, 4, 2, -1],
                           [-1, 1, 2, 1, -1],
                           [-1, -1, -1, -1, -1]])
    else:
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])

    filtered = ndimage.convolve(np_array, kernel)
    if average:
        filtered *= np.mean(np_array) / np.mean(filtered)

    return filtered


def butterworth_highpass_filter(image, d0=30, n=2):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    fft_image = np.fft.fft2(image)

    mask = 1 / (1 + ((np.sqrt((np.arange(rows)[:, np.newaxis] - crow) ** 2 +
                              (np.arange(cols)[np.newaxis, :] - ccol) ** 2)) / d0) ** (2 * n))

    # 应用滤波器

    fft_image_shifted = np.fft.fftshift(fft_image)
    fft_image_filtered = fft_image_shifted * (1 - mask)
    result_image = np.fft.ifft2(np.fft.ifftshift(fft_image_filtered)).real

    return result_image


def exponential_highpass_filter(image, d0=30):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # 创建指数高通滤波器
    mask = 1 - np.exp(-((np.arange(rows)[:, np.newaxis] - crow) ** 2 +
                        (np.arange(cols)[np.newaxis, :] - ccol) ** 2) / (d0 ** 2))
    # mask[256, 256] = 1
    # plot_parallel(
    #     a=mask
    # )
    # 应用滤波器
    fft_image = np.fft.fft2(image)
    fft_image_shifted = np.fft.fftshift(fft_image)
    fft_image_filtered = fft_image_shifted * mask
    result_image = np.fft.ifft2(np.fft.ifftshift(fft_image_filtered)).real

    # result_image *= np.mean(image) / np.mean(result_image)
    return result_image


def exponential_lowpass_filter(image, d0=30):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # 创建指数高通滤波器
    mask = np.exp(-((np.arange(rows)[:, np.newaxis] - crow) ** 2 +
                    (np.arange(cols)[np.newaxis, :] - ccol) ** 2) / (2 * d0 ** 2))

    # 应用滤波器
    fft_image = np.fft.fft2(image)
    fft_image_shifted = np.fft.fftshift(fft_image)
    fft_image_filtered = fft_image_shifted * mask
    result_image = np.fft.ifft2(np.fft.ifftshift(fft_image_filtered)).real

    # result_image *= np.mean(image) / np.mean(result_image)
    return result_image


def haar_wavelet_transform(image, level=1):
    """
    Apply Haar wavelet transform to a grayscale or color image.

    Parameters:
        image (numpy.ndarray): Input image, can be grayscale or color.
        level (int): Number of levels of the wavelet transform.

    Returns:
        numpy.ndarray: Transformed image.
    """

    def haar_wavelet_transform_single_channel(channel, level):
        h, w = channel.shape
        output = np.copy(channel).astype(float)

        for _ in range(level):
            temp = np.zeros_like(output)
            # Horizontal transform
            for i in range(h):
                for j in range(0, w, 2):
                    sum_val = (output[i, j] + output[i, j + 1]) / 2
                    diff_val = (output[i, j] - output[i, j + 1]) / 2
                    temp[i, j // 2] = sum_val
                    temp[i, j // 2 + w // 2] = diff_val

            # Vertical transform
            for i in range(0, h, 2):
                for j in range(w):
                    sum_val = (temp[i, j] + temp[i + 1, j]) / 2
                    diff_val = (temp[i, j] - temp[i + 1, j]) / 2
                    output[i // 2, j] = sum_val
                    output[i // 2 + h // 2, j] = diff_val

            h //= 2
            w //= 2

        return np.clip(output / 255, 0, 1)

    image *= 255
    if len(image.shape) == 2:  # Grayscale image
        return haar_wavelet_transform_single_channel(image, level)
    elif len(image.shape) == 3:  # Color image
        channels = cv2.split(image)
        transformed_channels = [haar_wavelet_transform_single_channel(ch, level) for ch in channels]
        return cv2.merge(transformed_channels)
    else:
        raise ValueError("Unsupported image format")


def uniform_lowpass_filter(image, r=64):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # 创建指数高通滤波器
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - r // 2:crow + r // 2, ccol - r // 2:ccol + r // 2] = 1

    # 应用滤波器
    fft_image = np.fft.fft2(image)
    fft_image_shifted = np.fft.fftshift(fft_image)
    fft_image_filtered = fft_image_shifted * mask

    result_image = np.fft.ifft2(np.fft.ifftshift(fft_image_filtered)).real

    # result_image *= np.mean(image) / np.mean(result_image)
    return result_image
