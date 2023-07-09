import sys
from typing import List, Tuple

from detectron2.config import get_cfg
from detectron2.data import Metadata
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer
import numpy as np
import torch

# TODO: Avoid adding third party library paths
sys.path.append("third_party/Detic/third_party/CenterNet2")
from centernet.config import add_centernet_config

sys.path.append("third_party/Detic")
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.predictor import get_clip_embeddings


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
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = False

    cfg.freeze()
    return cfg


class DeticPredictor:
    """Class to perform object detection using Detic."""

    def __init__(self):
        cfg = setup_cfg()
        self.predictor = DefaultPredictor(cfg)

        self.cpu_device = torch.device("cpu")

    def predict(
        self, image: np.ndarray, class_names: List[str]
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Runs object detection on the input image.

        Args:
            image: a color image in BGR channel order.
            custom_vocabulary: a list containing class names to detect.

        Returns:
            List of names of detected objects.
            Mask prediction.
            Visualized image output.
        """
        # Create classifier for given class names
        metadata = Metadata()
        metadata.thing_classes = class_names
        classifier = get_clip_embeddings(metadata.thing_classes)
        num_classes = len(metadata.thing_classes)
        reset_cls_test(self.predictor.model, classifier, num_classes)

        # Run prediction
        predictions = self.predictor(image)

        # Parse predictions
        instances = predictions["instances"].to(self.cpu_device)

        classes = instances.pred_classes.numpy()
        class_names = metadata.thing_classes
        names = [class_names[i] for i in classes]

        masks = instances.pred_masks.detach().numpy()
        masks = masks.astype(np.uint8) * 255

        return names, masks, self.visualize(image, metadata, instances)

    def visualize(
        self, image: np.ndarray, metadata: Metadata, instances: Instances
    ) -> np.ndarray:
        visualizer = Visualizer(image, metadata)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)
        return vis_output.get_image()
