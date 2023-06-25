from typing import List

import numpy as np


def create_camera_matrix(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Creates camera matrix from focal lengths and principal points.

    Args:
        fx: Focal length of the camera in the x direction.
        fy: Focal length of the camera in the y direction.
        cx: X-coordinate of the principal point.
        cy: Y-coordinate of the principal point.

    Returns:
        Camera matrix of shape (3, 3)
    """
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def create_dist_coeffs(dist_coeffs: List[float]) -> np.ndarray:
    """Creates numpy distortion coefficients

    Args:
        dist_coeffs: Distortion coefficients in a list.

    Returns:
        Distortion coefficients in a numpy array.
    """
    return np.array(dist_coeffs, dtype=np.float32)
