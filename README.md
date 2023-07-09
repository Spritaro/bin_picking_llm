# Bin picking LLM

This repository contains code to perform bin picking with a RealSense camera, a Dobot Magician robotic arm, and ChatGPT.

## Setup

To set up the project, follow these steps:

1. Clone the repository and submodules.

   ```sh
   git clone --recursive https://github.com/Spritaro/bin_picking_llm.git
   ```

   or

   ```sh
   git clone https://github.com/Spritaro/bin_picking_llm.git
   git submodule update --init --recursive
   ```

1. Create and activate virtual environment using venv.

    ```sh
    cd bin_picking_llm
    python3 -m venv venv
    source venv/bin/activate
    ```

1. Install PyTorch.

    ```sh
    pip3 install --upgrade pip setuptools wheel
    pip3 install torch torchvision
    ```

    If you are using Jetson, see [Installing PyTorch for Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) and [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048). Only PyTorch binary is available, so you may need to build torchvision yourself.

1. Install dependencies.

    ```sh
    pip3 install --no-build-isolation -e .
    ```

    If you are using Jetson with JetPack 4.x, install librealsense2 with apt and pyrealsense2 with pip. If JetPack 5.x, you will need to build librealsense2 and pyrealsense2.

1. Download Detic model and install Detic dependencies.

    ```sh
    cd third_party/Detic
    mkdir models
    cd models
    wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_R18_640b32_4x_ft4x_max-size.pth
    ```

    ```sh
    pip3 install -r requirements.txt
    cd third_party/Deformable-DETR/models/ops
    ./make.sh
    ```

1. Connect the camera and the robot to the computer.

    Permission for the serial device may need to be modified.

    ```sh
    sudo chmod a+rw /dev/ttyUSB0
    ```

## Calibration

Hand-eye calibration is necessary for bin picking. Follow these steps.

1. Place a checkerboard in front of the camera and the robot. It should be placed in a way that the robotic arm is able to reach all corners of the checkerboard.

1. Run calibration script and launch calibration window. Adjust parameters to reflect the actual checkerboard size.

    ```sh
    python3 scripts/calibrate_camera_robot_base.py --column 10 --row 7 --square-size 19.09
    ```

1. Calibrate camera base pose by pressing ```c``` key. A calibration file should be saved to ```outputs/camera_base.npy```

1. Calibrate robot base pose by following these steps.
    1. Move the arm to the origin of the checkerboard (shown by axis on the window), then press ```Enter```.
    1. From the current position, move the arm in the X axis (red) direction, touch the outmost corner, then press ```Enter```.
    1. From the current position, move the arm in the Y axis (green) direction, touch the outmost corner, then press ```Enter```.
    1. From the current position, move the arm in the negative X axis direction, touch the outmost corner, then press ```Enter```.
    1. A calibration file should be automatically saved to ```outputs/robot_base.py```.

1. Quit the script by pressing ```q``` key.

## Usage

1. Create OpenAI API Key and set it to the environment variable.

    ```sh
    export OPENAI_API_KEY="your_api_key"
    ```

1. Run the following script.

    ```sh
    python3 scripts/move_robot_with_llm.py
    ```

1. Enter prompts such as "Find a Toothbrush and move your arm to its position".
