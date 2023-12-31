from typing import List, Optional, Tuple

import cv2
import numpy as np

from .utils import create_camera_matrix, create_dist_coeffs


def compute_3d_position(
    depth: np.ndarray, mask: np.ndarray, camera_matrix: np.ndarray
) -> Optional[np.ndarray]:
    """Computes the 3D position of an object given depth and mask images.

    Args:
        depth: Depth image of shape (height, width) in mm.
        mask: Mask image of shape (height, width).
        camera_matrix: Camera matrix of shape (3, 3).

    Returns:
        3D position of the object in camera coordinates in mm.
    """
    # The point cloud unit will be the same as depth only when depth is float
    depth = depth.astype(np.float32)

    point_cloud = cv2.rgbd.depthTo3d(depth=depth, K=camera_matrix, mask=mask)
    if np.all(np.isnan(point_cloud)):
        return None

    position = np.nanmean(point_cloud[0], axis=0)
    if np.any(np.isnan(position)):
        return None

    return position


def apply_transform(position: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Applies affine transform matrix to a 3D position.

    Args:
        position: The 3D position to transform.
        transform: The affine transform matrix of shape (4, 4).

    Returns:
        The transformed 3D position.
    """
    # Append 1 to the position vector
    position = np.append(position, 1)

    # Apply transform
    position = np.dot(transform, position)

    # Extract the transformed position
    return position[:3]


class PoseCalculator:
    """Class for calculating 3D positions using depth and mask images with
    camera intrinsics.

    Args:
        fx: Focal length of the camera in the x direction.
        fy: Focal length of the camera in the y direction.
        cx: X-coordinate of the principal point.
        cy: Y-coordinate of the principal point.
        dist_coeffs: Distortion coefficients of the camera.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dist_coeffs: List[float],
        camera_to_base: np.ndarray,
        base_to_robot: np.ndarray,
    ):
        self._camera_matrix = create_camera_matrix(fx, fy, cx, cy)
        self._dist_coeffs = create_dist_coeffs(dist_coeffs)

        self._camera_to_robot = np.dot(base_to_robot, camera_to_base)
        # self._camera_to_base = camera_to_base
        # self._base_to_robot = base_to_robot

        self._rvec = np.zeros(3, dtype=np.float32)
        self._tvec = np.zeros(3, dtype=np.float32)

    def calculate(
        self,
        depth: np.ndarray,
        masks: List[np.ndarray],
        color: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """
        Calculate the 3D poses from depth and mask images.

        Args:
            depth: Depth image.
            masks: List of mask images.
            color: Color image. Defaults to None.

        Returns:
            List of computed 3D poses and an optional.
            Color image with 3D center positions drawn.
        """
        pos_color = color.copy() if color is not None else None

        positions = []
        for mask in masks:
            position = compute_3d_position(depth, mask, self._camera_matrix)
            # print(f"before {position}")

            if position is None:
                positions.append(None)
                continue

            if color is not None:
                # Draw 3D center position
                image_point, _ = cv2.projectPoints(
                    position,
                    self._rvec,
                    self._tvec,
                    self._camera_matrix,
                    self._dist_coeffs,
                )
                center = image_point.flatten().astype(np.int32)
                cv2.circle(pos_color, center, radius=3, color=(0, 0, 255))

            position = apply_transform(position, self._camera_to_robot)
            # position = apply_transform(position, self._camera_to_base)
            # print(f"middle {position}")
            # position = apply_transform(position, self._base_to_robot)
            # print(f"after {position}")
            # print("")
            positions.append(position)

        return positions, pos_color
