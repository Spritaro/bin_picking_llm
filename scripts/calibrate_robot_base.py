"""Script to estimate camera to checkerboard transformation"""

import cv2

from bin_picking_llm.robot import Dobot

from bin_picking_llm.calibration import RobotBaseCalibrator
from bin_picking_llm.camera import RealSenseCamera
from options import get_command_line_arguments


def main():
    args = get_command_line_arguments()

    with Dobot() as robot, RealSenseCamera() as camera:
        calibrator = RobotBaseCalibrator(
            rows=args.rows,
            cols=args.columns,
            square_size=args.square_size,
        )

        while True:
            image, _ = camera.get_images()
            if image is None:
                continue

            cv2.imshow("Pose Estimation", image)
            key = cv2.waitKey(1)

            if key & 0xFF == ord("q"):
                break

            if key & 0xFF == 27:  # ESC key
                calibrator.clear_points()
                print("Cleared points")

            if key & 0xFF == 13:  # Enter key
                x, y, z = robot.get_current_position()
                calibrator.add_point(x, y, z)
                print(f"Added point: {x} {y} {z}")

                if calibrator.is_ready():
                    matrix = calibrator.calc_transform_matrix()
                    print(matrix)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
