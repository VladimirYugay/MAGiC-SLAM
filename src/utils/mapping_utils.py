""" This module contains utility functions for mapping. The majority of the functions are taken from:
    https://github.com/VladimirYugay/Gaussian-SLAM/blob/main/src/utils/mapper_utils.py
"""
import cv2
import numpy as np


def sample_pixels_based_on_gradient(image: np.ndarray, num_samples: int) -> np.ndarray:
    """ Samples pixel indices based on the gradient magnitude of an image.
    Args:
        image: The image from which to sample pixels.
        num_samples: The number of pixels to sample.
    Returns:
        Indices of the sampled pixels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize the gradient magnitude to create a probability map
    prob_map = grad_magnitude / np.sum(grad_magnitude)

    # Flatten the probability map
    prob_map_flat = prob_map.flatten()

    # Sample pixel indices based on the probability map
    sampled_indices = np.random.choice(prob_map_flat.size, size=num_samples, p=prob_map_flat)
    return sampled_indices.T


def geometric_edge_mask(rgb_image: np.ndarray, dilate: bool = True, RGB: bool = False) -> np.ndarray:
    """ Computes an edge mask for an RGB image using geometric edges.
    Args:
        rgb_image: The RGB image.
        dilate: Whether to dilate the edges.
        RGB: Indicates if the image format is RGB (True) or BGR (False).
    Returns:
        An edge mask of the input image.
    """
    # Convert the image to grayscale as Canny edge detection requires a single channel image
    gray_image = cv2.cvtColor(
        rgb_image, cv2.COLOR_BGR2GRAY if not RGB else cv2.COLOR_RGB2GRAY)
    if gray_image.dtype != np.uint8:
        gray_image = gray_image.astype(np.uint8)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200, apertureSize=3, L2gradient=True)
    # Define the structuring element for dilation, you can change the size for a thicker/thinner mask
    if dilate:
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def create_point_cloud(image: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Creates a point cloud from an image, depth map, camera intrinsics, and pose.

    Args:
        image: The RGB image of shape (H, W, 3)
        depth: The depth map of shape (H, W)
        intrinsics: The camera intrinsic parameters of shape (3, 3)
        pose: The camera pose of shape (4, 4)
    Returns:
        A point cloud of shape (N, 6) with last dimension representing (x, y, z, r, g, b)
    """
    height, width = depth.shape
    # Create a mesh grid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    # Convert pixel coordinates to camera coordinates
    x = (u - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    z = depth
    # Stack the coordinates together
    points = np.stack((x, y, z, np.ones_like(z)), axis=-1)
    # Reshape the coordinates for matrix multiplication
    points = points.reshape(-1, 4)
    # Transform points to world coordinates
    posed_points = pose @ points.T
    posed_points = posed_points.T[:, :3]
    # Flatten the image to get colors for each point
    colors = image.reshape(-1, 3)
    # Concatenate posed points with their corresponding color
    point_cloud = np.concatenate((posed_points, colors), axis=-1)

    return point_cloud
