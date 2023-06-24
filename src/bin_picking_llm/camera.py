"""Module for accessing RealSense camera."""

from typing import Dict, Tuple, Optional

import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    """Class to access RealSense camera."""

    def __init__(self):
        """Initializes the RealSenseCamera object."""
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._align = rs.align(rs.stream.color)
        self._config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self._config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    def __enter__(self):
        self._pipeline.start(self._config)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._pipeline.stop()

    def get_intrinsics(self) -> Dict:
        """Retrieves camera intrinsic parameters for both color and depth.

        Returns:
            A dictionary containing the camera intrinsic parameters.
                The keys are 'depth' and 'color', and the values are dict with
                    the following keys:
                - 'width': width of the image in pixels
                - 'height': height of the image in pixels
                - 'fx': focal length in the x direction
                - 'fy': focal length in the y direction
                - 'cx': x-coordinate of the principal point
                - 'cy': y-coordinate of the principal point
                - 'dist_coeff': distortion coefficients (k1, k2, p1, p2, k3)
        """
        intrinsics = {}
        frames = self._pipeline.wait_for_frames()
        aligned_frames = self._align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if color_frame:
            color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

            intrinsics["color"] = {
                "width": color_intrinsics.width,
                "height": color_intrinsics.height,
                "fx": color_intrinsics.fx,
                "fy": color_intrinsics.fy,
                "cx": color_intrinsics.ppx,
                "cy": color_intrinsics.ppy,
                "dist_coeffs": color_intrinsics.coeffs,
            }

        if depth_frame:
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            intrinsics["depth"] = {
                "width": depth_intrinsics.width,
                "height": depth_intrinsics.height,
                "fx": depth_intrinsics.fx,
                "fy": depth_intrinsics.fy,
                "cx": depth_intrinsics.ppx,
                "cy": depth_intrinsics.ppy,
                "dist_coeffs": depth_intrinsics.coeffs,
            }

        return intrinsics

    def get_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Retrieves the latest color and depth images from the camera.

        Returns:
            The color image or None if not available.
            The depth image or None if not available
        """
        frames = self._pipeline.wait_for_frames()
        aligned_frames = self._align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data()) if color_frame else None
        depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None

        return color_image, depth_image
