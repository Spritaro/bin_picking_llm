"""Script to calibrate camera base and robot base"""

import argparse

import cv2
import numpy as np

from bin_picking_llm.robot import Dobot

from bin_picking_llm import config
from bin_picking_llm.calibration import CameraBaseCalibrator
from bin_picking_llm.calibration import RobotBaseCalibrator
from bin_picking_llm.camera import RealSenseCamera


def get_args():
    parser = argparse.ArgumentParser(
        description="RealSense Camera Checkerboard Detection"
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=10,
        help="Number of columns in the checkerboard pattern",
    )
    parser.add_argument(
        "--rows", type=int, default=7, help="Number of rows in the checkerboard pattern"
    )
    parser.add_argument(
        "--square-size", type=float, default=19.09, help="Size of each square in mm"
    )
    return parser.parse_args()


def main():
    args = get_args()

    camera_matrix = None

    with RealSenseCamera() as camera, Dobot() as robot:
        intrinsics = camera.get_intrinsics()
        print(intrinsics)

        camera_calibrator = CameraBaseCalibrator(
            fx=intrinsics["color"]["fx"],
            fy=intrinsics["color"]["fy"],
            cx=intrinsics["color"]["cx"],
            cy=intrinsics["color"]["cy"],
            dist_coeffs=intrinsics["color"]["dist_coeffs"],
            rows=args.rows,
            cols=args.columns,
            square_size=args.square_size,
        )

        robot_calibrator = RobotBaseCalibrator(
            rows=args.rows,
            cols=args.columns,
            square_size=args.square_size,
        )

        while True:
            image, _ = camera.get_images()
            if image is None:
                continue

            rvecs, tvecs = camera_calibrator.estimate_pose(image)
            if rvecs is not None and tvecs is not None:
                camera_matrix = camera_calibrator.calc_transform_matrix(rvecs, tvecs)
                image = camera_calibrator.draw_axis(image, rvecs, tvecs)
            else:
                print("Checkerboard not found")

            cv2.imshow("Pose Estimation", image)
            key = cv2.waitKey(1)

            if key & 0xFF == ord("q"):
                break

            if key & 0xFF == ord("c"):
                print(camera_matrix)
                if camera_matrix is not None:
                    with open(config.CAMERA_BASE_PATH, "wb") as f:
                        np.save(f, camera_matrix)

            if key & 0xFF == 13:  # Enter key
                x, y, z = robot.get_current_position()
                robot_calibrator.add_point(x, y, z)
                print(f"Added point: {x} {y} {z}")

                if robot_calibrator.is_ready():
                    robot_matrix = robot_calibrator.calc_transform_matrix()
                    print(robot_matrix)
                    if robot_matrix is not None:
                        with open(config.ROBOT_BASE_PATH, "wb") as f:
                            np.save(f, robot_matrix)

            if key & 0xFF == 27:  # ESC key
                robot_calibrator.clear_points()
                print("Cleared points")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
