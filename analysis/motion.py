import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from scipy.interpolate import BSpline
from analysis.evaluation import compare_img
from scipy.ndimage import affine_transform, map_coordinates
from visualization.view_2D import *


def generate_grid_motion(image, num_views=20, max_translation=5, max_rotation=2.5):
    """
    Simulates 2-DoF motion artifacts (translation and rotation) on a 2D image.

    Parameters:
        image (numpy.ndarray): Input 2D array representing the image.
        num_views (int): Number of motion frames to simulate.
        max_translation (float): Maximum translation in pixels (both x and y directions).
        max_rotation (float): Maximum rotation in degrees.

    Returns:
        numpy.ndarray: The motion-artifact-affected image.
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be a 2D numpy array.")

    # Initialize the output image (accumulated motion blur)
    artifact_image = np.zeros_like(image, dtype=np.float32)

    # Generate random motion parameters for each view
    translations_x = np.random.uniform(-max_translation, max_translation, num_views)
    translations_y = np.random.uniform(-max_translation, max_translation, num_views)
    rotations = np.random.uniform(-max_rotation, max_rotation, num_views)

    # Apply motion to each view and accumulate the results
    for tx, ty, rot in zip(translations_x, translations_y, rotations):
        # Create affine transformation matrix
        theta = np.deg2rad(rot)  # Convert degrees to radians
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        translation_vector = np.array([tx, ty])

        # Combine rotation and translation into an affine transformation matrix
        affine_matrix = np.eye(3)
        affine_matrix[:2, :2] = rotation_matrix
        affine_matrix[:2, 2] = translation_vector

        # Apply the affine transformation to the image
        transformed_image = affine_transform(
            image,
            matrix=affine_matrix[:2, :2],
            offset=affine_matrix[:2, 2],
            order=1,
            mode='constant',
            cval=0.0
        )

        # Accumulate the transformed image
        artifact_image += transformed_image

    # Normalize the accumulated image
    artifact_image /= num_views

    return artifact_image


def simulate_non_rigid_motion(image, amplitude=(5, 5), frequency=(1, 1), phase=(0, 0)):
    """
    Simulates non-rigid motion (e.g., respiratory-like deformation) on a 2D image.

    Parameters:
        image (numpy.ndarray): Input 2D image.
        amplitude (tuple): Amplitude of deformation along x and y.
        frequency (tuple): Frequency of deformation along x and y.
        phase (tuple): Phase shift of the deformation along x and y.

    Returns:
        numpy.ndarray: Image with non-rigid motion applied.
    """
    # Generate grid coordinates
    y, x = np.arange(image.shape[0]), np.arange(image.shape[1])
    x_grid, y_grid = np.meshgrid(x, y)

    # Deformation field
    x_deform = amplitude[0] * np.sin(2 * np.pi * frequency[0] * (y_grid / image.shape[0]) + phase[0])
    y_deform = amplitude[1] * np.sin(2 * np.pi * frequency[1] * (x_grid / image.shape[1]) + phase[1])

    # Apply deformation to grid coordinates
    x_deformed = x_grid + x_deform
    y_deformed = y_grid + y_deform

    # Interpolate the image at the deformed coordinates
    deformed_image = map_coordinates(image, [y_deformed, x_deformed], order=1, mode='nearest')

    return deformed_image


def generate_nongrid_motion(image, num_views=10, amplitude=(5, 5), frequency=(2, 2)):
    """
    Generates a motion artifact by averaging multiple frames with motion applied.

    Parameters:
        image (numpy.ndarray): Input 2D image.
        num_views (int): Number of views (frames) to simulate.
        motion_type (str): Type of motion ("rigid" or "non_rigid").
        motion_params (dict): Parameters for the motion simulation.

    Returns:
        numpy.ndarray: Image with motion artifact applied.
    """
    frames = []
    for i in range(num_views):
        # Generate random phase for non-rigid motion
        phase = (
            np.random.uniform(0, 2 * np.pi),
            np.random.uniform(0, 2 * np.pi)
        )
        frame = simulate_non_rigid_motion(
            image,
            amplitude=amplitude,
            frequency=frequency,
            phase=phase
        )
        frames.append(frame)

    # Average all frames to simulate motion artifact
    motion_artifact_image = np.mean(frames, axis=0)
    return motion_artifact_image
