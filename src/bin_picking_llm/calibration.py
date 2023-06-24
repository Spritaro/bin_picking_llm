"""Module for calibration."""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class VisualPoseEstimator:
    """Class for estimating the pose of a checkerboard pattern in an image."""

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dist_coeffs: List[float],
        rows: int,
        cols: int,
        square_size: float,
    ):
        """Initializes the pose estimator object.

        Args:
            fx: Focal length of the camera in the x direction.
            fy: Focal length of the camera in the y direction.
            cx: X-coordinate of the principal point.
            cy: Y-coordinate of the principal point.
            dist_coeffs: Distortion coefficients of the camera.
            rows: Number of rows in the checkerboard pattern.
            cols: Number of columns in the checkerboard pattern.
            square_size: Size of each square in the checkerboard pattern.
        """
        self._camera_matrix = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
        )
        self._dist_coeffs = np.array(dist_coeffs, dtype=np.float32)

        self._rows = rows
        self._cols = cols
        self._square_size = square_size

        self._criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        self._object_points = np.zeros((self._cols * self._rows, 3), np.float32)
        self._object_points[:, :2] = np.mgrid[: self._cols, : self._rows].T.reshape(
            -1, 2
        )

    def estimate_pose(self, image) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimates the pose of the checkerboard pattern in the given image.

        Args:
            image: Input image containing the checkerboard pattern.

        Returns:
            Rotation vector representing the orientation of the checkerboard.
            Translation vector representing the position of the checkerboard.
                Scaled by the square size.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find corners of the checkerboard
        found, corners = cv2.findChessboardCorners(gray, (self._cols, self._rows), None)
        if not found:
            return None, None

        # Refine the corner positions
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self._criteria)

        # Estimate the pose of the checkerboard
        _, rvecs, tvecs, _ = cv2.solvePnPRansac(
            objectPoints=self._object_points,
            imagePoints=corners,
            cameraMatrix=self._camera_matrix,
            distCoeffs=self._dist_coeffs,
        )
        return rvecs, tvecs * self._square_size

    def calc_transform_matrix(self, rvecs: np.ndarray, tvecs: np.ndarray) -> np.ndarray:
        """
        Calculates the affine transformation matrix representing the pose of the checkerboard.

        Args:
            rvecs: Rotation vector representing the orientation of the checkerboard.
            tvecs: Translation vector representing the position of the checkerboard.
                Scaled by the square size.

        Returns:
            4x4 affine transformation matrix.

        """
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvecs)

        # Create affine transformation matrix
        transform_matrix = np.zeros((4, 4), dtype=np.float32)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3:4] = tvecs
        transform_matrix[3, 3] = 1
        return transform_matrix

    def draw_axis(
        self, image: np.ndarray, rvecs: np.ndarray, tvecs: np.ndarray
    ) -> np.ndarray:
        """
        Draws the coordinate axes on the input image.

        Args:
            image: Input image.
            rvecs: Rotation vector representing the orientation of the checkerboard.
            tvecs: Translation vector representing the position of the checkerboard.
                Scaled by the square size.

        Returns:
            np.ndarray: Image with the coordinate axes drawn.
        """
        return cv2.drawFrameAxes(
            image,
            self._camera_matrix,
            self._dist_coeffs,
            rvecs,
            tvecs,
            self._square_size * 3,
        )
