import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes
from PIL import Image
from scipy.ndimage import zoom


def get_bounding_box(np_array, target):
    loc = np.array(np.where(target > 0))
    x_min, x_max = np.min(loc[0]), np.max(loc[0])
    y_min, y_max = np.min(loc[1]), np.max(loc[1])
    z_min, z_max = np.min(loc[2]), np.max(loc[2])

    return np_array[x_min:x_max, y_min:y_max, z_min:z_max], \
           target[x_min:x_max, y_min:y_max, z_min:z_max]


def padding_into_512(np_array):
    h = 236 - int(np_array.shape[-1] / 2)

    new_array = np.zeros([512, 512, 512])
    new_array[:, :, h:h + np_array.shape[-1]] = np_array

    return new_array, h, np_array.shape[-1]


def rescale_95_intensity(image, low=0.5, high=99.5):
    # Calculate the 0.5th and 99.5th percentiles
    lower_percentile = np.percentile(image, low)
    upper_percentile = np.percentile(image, high)

    # Clip the intensity values to the range between the percentiles
    clipped_image = np.clip(image, lower_percentile, upper_percentile)

    # Rescale the clipped values to the range [0, 1]
    rescaled_image = (clipped_image - lower_percentile) / (upper_percentile - lower_percentile)

    return rescaled_image


def center_crop(image_array):
    # Get the dimensions of the image
    height, width = image_array.shape

    # Check if we need to resize
    if width < 512 or height < 512:
        # Calculate the scale factors
        scale_factor = max(512 / width, 512 / height)

        # Resize the image using PIL
        # image = Image.fromarray(image_array)
        # new_width = int(width * scale_factor)
        # new_height = int(height * scale_factor)
        # image = image.resize((new_width, new_height), Image.LANCZOS)
        image = zoom(image_array, zoom=scale_factor)
        # Convert back to a numpy array
        image_array = np.array(image)
        height, width = image_array.shape

    # Calculate cropping coordinates
    left = (width - 512) // 2
    top = (height - 512) // 2

    # Crop the center of the image
    cropped_image = image_array[top:top + 512, left:left + 512]

    return cropped_image
