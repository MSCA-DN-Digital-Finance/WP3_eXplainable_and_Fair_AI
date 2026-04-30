import pytest
import numpy as np


import sys
import os

# Get the absolute path to the 'src' directory
# This looks up two levels from the current test file
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(src_path)

from prediction.sample_builder import build_samples


testdata = [
    # Tensor is (Samples, Time Steps, Dimensions)


    # Case 0: 1 sample, 1 dimension
    (
        np.array([0, 1, 2, 3, 4, 5]),
        {'gap': 0, 'output_length': 1, 'input_length': 5},
        np.array([[[0], [1], [2], [3], [4]]]), # (1, 5, 1)
        np.array([[[5]]])                      # (1, 1, 1)
    ),
    # Case 1: 1 sample, 3 dimensions
    (
        np.arange(18).reshape(6, 3),
        {'gap': 0, 'output_length': 1, 'input_length': 5},
        np.arange(15).reshape(1, 5, 3),        # (1, 5, 3)
        np.array([[[15, 16, 17]]])             # (1, 1, 3)
    ),

    # Case 2: 3 samples, 2 dimensions
    (
        np.arange(10).reshape(5,2),
        {'gap': 0, 'output_length': 1, 'input_length': 2},
        np.array([[[0, 1], [2, 3]],
                  [[2, 3], [4, 5]],
                  [[4, 5], [6, 7]]]),        # (3, 2, 2)
        np.array([[[4, 5]],
                  [[6, 7]],
                  [[8, 9]]])             # (3, 1, 2)
    ),
    # Case 3: 1 sample, 1 dimension, 1 gap
    (
        np.array([0, 1, 2, 3, 4, 5]),
        {'gap': 1, 'output_length': 1, 'input_length': 4},
        np.array([[[0], [1], [2], [3]]]), # (1, 4, 1)
        np.array([[[5]]])                      # (1, 1, 1)
    )
]


@pytest.mark.parametrize("traj,params,x_val,y_val", testdata)
def test_sample_creation(traj, params, x_val, y_val ):

    x,y = build_samples(traj, **params)

    assert np.array_equal(x, x_val)
    assert np.array_equal(y, y_val)
