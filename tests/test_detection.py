
import numpy as np

from bin_picking_llm.predictor import DeticPredictor


def test_single_prediction():
    predictor = DeticPredictor()

    image = np.zeros((640, 480, 3), dtype=np.uint8)

    class_names = "person"

    names, predictions, vis_output = predictor.predict(image, class_names)

    assert isinstance(names, list)
    assert isinstance(predictions, np.ndarray)
    assert isinstance(vis_output, np.ndarray)


def test_multiple_predictions():
    predictor = DeticPredictor()

    image = np.zeros((640, 480, 3), dtype=np.uint8)

    class_names = "dog,cat"

    names, predictions, vis_output = predictor.predict(image, class_names)

    assert isinstance(names, list)
    assert isinstance(predictions, np.ndarray)
    assert isinstance(vis_output, np.ndarray)


def test_consecutive_predictions():
    predictor = DeticPredictor()

    image = np.zeros((640, 480, 3), dtype=np.uint8)

    class_names = "dog"
    names, predictions, vis_output = predictor.predict(image, class_names)

    class_names = "cat"
    names, predictions, vis_output = predictor.predict(image, class_names)

    assert isinstance(names, list)
    assert isinstance(predictions, np.ndarray)
    assert isinstance(vis_output, np.ndarray)
