import numpy as np

from bin_picking_llm import config
from bin_picking_llm.camera import RealSenseCamera
from bin_picking_llm.llm import ChatGPT
from bin_picking_llm.pose import PoseCalculator
from bin_picking_llm.predictor import DeticPredictor
from bin_picking_llm.robot import Dobot


functions = [
    {
        "name": "detect_objects",
        "description": "Runs object detection and returns the names and the 3D positions of the detected objects. The unit is in mm",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "move_robot_arm",
        "description": "Moves robot arm to specified position.",
        "parameters": {
            "type": "object",
            "properties": {
                "position": {
                    "type": "string",
                    "description": "A string representing a 3D position. The unit is in mm.",
                }
            },
            "required": ["position"],
        },
    },
    {
        "name": "grasp",
        "description": "Grasps an object at the robot arm position.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "release",
        "description": "Releases the robot gripper.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def main():
    with RealSenseCamera() as camera, Dobot() as robot:
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

        def detect_objects():
            print("Running object detection.")

            color, depth = camera.get_images()
            if color is None or depth is None:
                return "Failure"

            names, masks, _ = predictor.predict(color)
            positions, _ = pose_calculator.calculate(depth, masks)
            return [
                {
                    "name": name,
                    "position": position.tolist() if position is not None else None,
                }
                for name, position in zip(names, positions)
            ]

        def move_robot_arm(position):
            print(f"Moving robot arm to {position}.")

            position = [float(x.strip()) for x in position.split(",")]
            robot.move_to(*position, r=0)
            return "Success"

        def grasp():
            print("Grasping.")
            return "Success"

        def release():
            print("Releasing.")
            return "Success"

        function_map = {
            "detect_objects": detect_objects,
            "move_robot_arm": move_robot_arm,
            "grasp": grasp,
            "release": release,
        }

        while True:
            prompt = input("Prompt: ")

            llm = ChatGPT(functions, function_map)
            message = llm.chat(prompt)
            print(f"Response: {message}")


if __name__ == "__main__":
    main()
