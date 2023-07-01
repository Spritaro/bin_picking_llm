
import numpy as np

from bin_picking_llm.predictor import DeticPredictor


def test_predictor():
    predictor = DeticPredictor()

    image = np.zeros((640, 480, 3), dtype=np.uint8)

    predictions, vis_output = predictor.predict(image)

    assert isinstance(predictions, np.ndarray)
    assert isinstance(vis_output, np.ndarray)
