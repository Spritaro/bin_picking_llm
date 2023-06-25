import numpy as np
from bin_picking_llm.calibration import RobotBaseCalibrator


def test_robot_base_identity_transform():
    """Test identity transform for the robot base."""
    rows = 3
    cols = 4
    square_size = 10
    calibrator = RobotBaseCalibrator(rows, cols, square_size)

    # Add calibration points
    calibrator.add_point(0, 0, 0)
    calibrator.add_point(rows * square_size, 0, 0)
    calibrator.add_point(rows * square_size, cols * square_size, 0)
    calibrator.add_point(0, cols * square_size, 0)

    # Check if the calibrator is ready
    assert calibrator.is_ready()

    # Calculate the transform matrix
    transform_matrix = calibrator.calc_transform_matrix()

    print(transform_matrix)

    # Check the result
    assert isinstance(transform_matrix, np.ndarray)
    assert transform_matrix.shape == (4, 4)
    assert np.allclose(transform_matrix[0, :], [1, 0, 0, 0])
    assert np.allclose(transform_matrix[1, :], [0, 1, 0, 0])
    assert np.allclose(transform_matrix[3, :], [0, 0, 0, 1])


def test_robot_base_not_ready():
    """Test whether the calibrator correctly recognizes when not ready."""
    rows = 3
    cols = 3
    square_size = 10
    calibrator = RobotBaseCalibrator(rows, cols, square_size)

    # Add only 3 calibration points (less than required)
    calibrator.add_point(0, 0, 0)
    calibrator.add_point(0, square_size, 0)
    calibrator.add_point(square_size, square_size, 0)

    # Check if the calibrator is not ready
    assert not calibrator.is_ready()

    # Calculate the transform matrix
    transform_matrix = calibrator.calc_transform_matrix()

    # Check the result (should be None)
    assert transform_matrix is None
