from collections import namedtuple
import os.path
import sys
from typing import Optional, Tuple

import cv2
from detectron2.config import get_cfg
from detectron2.utils.visualizer import VisImage
import numpy as np
import torch

# TODO: Avoid adding third party library paths
sys.path.append("third_party/Detic/third_party/CenterNet2")
from centernet.config import add_centernet_config

sys.path.append("third_party/Detic")
from detic.config import add_detic_config
from detic.predictor import BUILDIN_CLASSIFIER, VisualizationDemo


def compute_3d_position(
    depth: np.ndarray, mask: np.ndarray, camera_matrix: np.ndarray
) -> Optional[np.ndarray]:
    """Compute the 3D position of an object given depth and mask images.

    Args:
        depth: Depth image of shape (height, width) in mm.
        mask: Mask image of shape (height, width).
        camera_matrix: Camera matrix of shape (3, 3).

    Returns:
        3D position of the object in camera coordinates.
    """
    point_cloud = cv2.rgbd.depthTo3d(depth=depth, K=camera_matrix, mask=mask)
    if np.all(np.isnan(point_cloud)):
        return None

    position = np.nanmean(point_cloud[0], axis=0)
    if np.any(np.isnan(position)):
        return None

    return position


def setup_cfg():
    """Setup cfg object for creating Detic predictor."""
    cfg = get_cfg()

    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(
        "third_party/Detic/configs/Detic_LCOCOI21k_CLIP_R18_640b32_4x_ft4x_max-size.yaml"
    )

    # Set score_threshold for builtin models
    threshold = 0.5
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = threshold

    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = (
        "third_party/Detic/datasets/metadata/lvis_v1_train_cat_info.json"
    )

    cfg.MODEL.WEIGHTS = (
        "third_party/Detic/models/Detic_LCOCOI21k_CLIP_R18_640b32_4x_ft4x_max-size.pth"
    )

    # Predict all classes
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True

    cfg.freeze()
    return cfg


class DeticPredictor:
    """Class to perform object detection using Detic."""

    def __init__(self):
        Args = namedtuple("Args", "vocabulary, custom_vocabulary")
        args = Args(vocabulary="lvis", custom_vocabulary="")

        # NOTE: Hacky workaround to fix classifier path
        BUILDIN_CLASSIFIER[args.vocabulary] = os.path.join(
            "third_party/Detic", BUILDIN_CLASSIFIER[args.vocabulary]
        )

        cfg = setup_cfg()

        self.demo = VisualizationDemo(cfg, args)

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, VisImage]:
        """Runs object detection on the input image.

        Args:
            image: a color image in BGR channel order.

        Returns:
            Mask prediction.
            Visualized image output.
        """
        predictions, vis_output = self.demo.run_on_image(image)

        masks = predictions["instances"].pred_masks.detach().cpu().numpy()
        masks = masks.astype(np.uint8) * 255

        return masks, vis_output
