"""Module for calibration."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .utils import create_camera_matrix, create_dist_coeffs


class CameraBaseCalibrator:
    """Class for calibrating the pose of a checkerboard pattern with respect
    to a camera."""

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
        """Initializes the camera base calibrator object.

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
        self._camera_matrix = create_camera_matrix(fx, fy, cx, cy)
        self._dist_coeffs = create_dist_coeffs(dist_coeffs)

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


class RobotBaseCalibrator:
    """Class for calibrating the pose of a robot with respect to a
    checkerboard."""

    def __init__(
        self,
        rows: int,
        cols: int,
        square_size: float,
    ):
        """Initializes robot base calibrator object.

        Args:
            rows: Number of rows in the checkerboard pattern.
            cols: Number of columns in the checkerboard pattern.
            square_size: Size of each square in the checkerboard
            pattern.
        """
        self._board_height = rows * square_size
        self._board_width = cols * square_size
        self.clear_points()

    def clear_points(self) -> None:
        """Clears the previously added calibration points."""
        self._points = []

    def add_point(self, x: float, y: float, z: float) -> None:
        """Adds a calibration point.

        Args:
            x: X-coordinate of the calibration point.
            y: Y-coordinate of the calibration point.
            z: Z-coordinate of the calibration point.
        """
        self._points.append([x, y, z])

    def is_ready(self) -> bool:
        """Checks if enough calibration points have been added.

        Returns:
            True if enough calibration points have been added, False
            otherwise.
        """
        return len(self._points) == 4

    def calc_transform_matrix(self) -> Optional[np.ndarray]:
        """Calculates the affine transformation matrix for the robot base.

        Returns:
            The affine transformation matrix, or None if the required number
            of calibration points is not reached or ifoutliers are detected.
        """
        if not self.is_ready():
            return None

        board_points = np.array(
            [
                [0, 0, 0],
                [self._board_height, 0, 0],
                [self._board_height, self._board_width, 0],
                [0, self._board_width, 0],
            ],
            dtype=np.float32,
        )

        robot_points = np.array(self._points, dtype=np.float32)

        # Create affine transformation matrix
        _, out, inliers = cv2.estimateAffine3D(src=board_points, dst=robot_points)

        # All 4 points should be inliers
        assert all(map(lambda x: x == 1, inliers))

        affine_transform_matrix = np.zeros((4, 4), dtype=np.float32)
        affine_transform_matrix[:3, :] = out
        affine_transform_matrix[3, 3] = 1
        return affine_transform_matrix
