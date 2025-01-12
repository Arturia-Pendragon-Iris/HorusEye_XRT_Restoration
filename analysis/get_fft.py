import numpy as np
import cv2
import matplotlib.pyplot as plt
from visualization.view_2D import plot_parallel


def do_fft(img, return_angle=False):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # fshift[254:259, 254:259] = 0
    fshift = np.log(fshift + 1)

    if return_angle:
        angle = np.angle(f)
        return np.abs(fshift), angle
    else:
        return np.abs(fshift)


