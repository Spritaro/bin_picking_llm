import cv2
import numpy as np

from bin_picking_llm import config
from bin_picking_llm.camera import RealSenseCamera
from bin_picking_llm.pose import PoseCalculator
from bin_picking_llm.predictor import DeticPredictor


def main():
    with RealSenseCamera() as camera:
        intrinsics = camera.get_intrinsics()
        pose_calculator = PoseCalculator(
            fx=intrinsics["depth"]["fx"],
            fy=intrinsics["depth"]["fy"],
            cx=intrinsics["depth"]["cx"],
            cy=intrinsics["depth"]["cy"],
            dist_coeffs=intrinsics["depth"]["dist_coeffs"],
            camera_to_base=np.load(config.CAMERA_BASE_PATH),
            base_to_robot=np.load(config.ROBOT_BASE_PATH),
        )

        predictor = DeticPredictor()

        while True:
            color, depth = camera.get_images()
            if color is None or depth is None:
                continue

            # 2D object detection
            names, masks, vis_output = predictor.predict(color)
            print(names)

            # 3D pose calculation
            _, pos_output = pose_calculator.calculate(depth, masks, color)

            cv2.imshow(
                "Detection Result",
                cv2.cvtColor(vis_output, cv2.COLOR_RGB2BGR),
            )
            cv2.imshow("Position Estimation", pos_output)
            key = cv2.waitKey(0)

            if key & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()
