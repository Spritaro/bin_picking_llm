"""Script to calibrate transformation of camera base to checkerboard"""

import cv2
import numpy as np

from bin_picking_llm import config
from bin_picking_llm.camera import RealSenseCamera
from bin_picking_llm.calibration import CameraBaseCalibrator
from options import get_command_line_arguments


def main():
    args = get_command_line_arguments()

    matrix = None

    with RealSenseCamera() as camera:
        intrinsics = camera.get_intrinsics()
        print(intrinsics)

        calibrator = CameraBaseCalibrator(
            fx=intrinsics["color"]["fx"],
            fy=intrinsics["color"]["fy"],
            cx=intrinsics["color"]["cx"],
            cy=intrinsics["color"]["cy"],
            dist_coeffs=intrinsics["color"]["dist_coeffs"],
            rows=args.rows,
            cols=args.columns,
            square_size=args.square_size,
        )

        while True:
            image, _ = camera.get_images()
            if image is None:
                continue

            rvecs, tvecs = calibrator.estimate_pose(image)
            if rvecs is not None and tvecs is not None:
                matrix = calibrator.calc_transform_matrix(rvecs, tvecs)
                image = calibrator.draw_axis(image, rvecs, tvecs)
            else:
                print("Checkerboard not found")

            cv2.imshow("Pose Estimation", image)
            key = cv2.waitKey(1)

            if key & 0xFF == ord("q"):
                break

            if key & 0xFF == 13:  # Enter key
                print(matrix)
                if matrix is not None:
                    np.save(config.CAMERA_BASE_PATH, matrix)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
