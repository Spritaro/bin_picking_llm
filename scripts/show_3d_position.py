import cv2
import numpy as np

from bin_picking_llm.camera import RealSenseCamera
from bin_picking_llm.detection import compute_3d_position, DeticPredictor
from bin_picking_llm.utils import create_camera_matrix, create_dist_coeffs


def main():
    with RealSenseCamera() as camera:
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)

        intrinsics = camera.get_intrinsics()
        camera_matrix = create_camera_matrix(
            fx=intrinsics["depth"]["fx"],
            fy=intrinsics["depth"]["fy"],
            cx=intrinsics["depth"]["cx"],
            cy=intrinsics["depth"]["cy"],
        )
        dist_coeffs = create_dist_coeffs(intrinsics["depth"]["dist_coeffs"])

        predictor = DeticPredictor()

        while True:
            color, depth = camera.get_images()
            if color is None or depth is None:
                continue

            # 2D object detection
            masks, vis_output = predictor.predict(color)

            # 3D object detection
            pos_color = color.copy()
            for mask in masks:
                position = compute_3d_position(depth, mask, camera_matrix)
                if position is None:
                    continue
                print(position)

                # Draw 3D center position
                image_point, _ = cv2.projectPoints(
                    position, rvec, tvec, camera_matrix, dist_coeffs
                )
                center = image_point.flatten().astype(np.int32)
                cv2.circle(pos_color, center, radius=3, color=(0, 0, 255))

            cv2.imshow(
                "Detection Result",
                cv2.cvtColor(vis_output.get_image(), cv2.COLOR_RGB2BGR),
            )
            cv2.imshow("Position Estimation", pos_color)
            key = cv2.waitKey(0)

            if key & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()
