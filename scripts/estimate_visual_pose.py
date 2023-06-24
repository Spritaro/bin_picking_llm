"""Script to estimate camera to checkerboard transformation"""

import argparse

import cv2

from bin_picking_llm.camera import RealSenseCamera
from bin_picking_llm.calibration import VisualPoseEstimator


def main():
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
    args = parser.parse_args()

    matrix = None

    camera = RealSenseCamera()
    camera.start()

    try:
        intrinsics = camera.get_intrinsics()
        print(intrinsics)

        estimator = VisualPoseEstimator(
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

            rvecs, tvecs = estimator.estimate_pose(image)
            if rvecs is not None and tvecs is not None:
                matrix = estimator.calc_transform_matrix(rvecs, tvecs)
                image = estimator.draw_axis(image, rvecs, tvecs)
            else:
                print("Checkerboard not found")

            cv2.imshow("Pose Estimation", image)
            key = cv2.waitKey(1)

            if key & 0xFF == ord("q"):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print(matrix)


if __name__ == "__main__":
    main()
