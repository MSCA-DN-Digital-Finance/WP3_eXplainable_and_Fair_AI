"""
This file contains unit tests for the param_stats functions in the `analysis.param_stats` module.

"""


import pytest
import numpy as np
import sys
import os


# Get the absolute path to the 'src' directory
# This looks up two levels from the current test file
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(src_path)

from analysis.param_stats import prob_positive, beta_hat, average_power_spectrum

################### Tests for prob_positive function ###################
prob_positive_testdata = [
    # Case 0: All positive values
    (
        {"yhat": np.array([1.0, 2.0, 3.0, 4.0])}, # input run
        1.0                            # expected probability
    ),
    # Case 1: All negative values
    (
        {"yhat": np.array([-1.0, -2.0, -3.0, -4.0])}, # input run
        0.0                            # expected probability
    ),
    # Case 2: Mixed values
    (
        {"yhat": np.array([-1.0, 1.0, -2.0, 2.0])}, # input run
        0.5                            # expected probability
    ),
    # Case 3: Empty trajectory
    (
        {"yhat": np.array([])}, # input run
        ValueError # raises ValueError due to empty input
    ),
    # Case 4: Wrong data type (string instead of numeric)
    (
        {"yhat": ["a", "b", "c"]}, # input run
        ValueError # raises ValueError due to invalid data type
    )
]
@pytest.mark.parametrize("input_run,expected_output", prob_positive_testdata)
def test_prob_positive(input_run, expected_output):
    """
    Tests the `prob_positive` function to ensure it correctly computes 
    the probability of positive values.
    """
    # Check if we expect a ValueError class or instance
    if expected_output == ValueError or isinstance(expected_output, ValueError):
        with pytest.raises(ValueError):
            prob_positive(input_run)
    else:
        assert prob_positive(input_run) == expected_output


#################### Tests for beta_hat function ###################
beta_hat_testdata = [

    # Case 0: Simple case with positive correlation
    (
        {
            "x": np.array([1.0, 2.0, 3.0, 4.0]),
            "yhat": np.array([1.0, 2.0, 3.0, 4.0]),
            "t_idx": np.array([0, 1, 2, 3])
        },
        1.0 # expected beta_hat
    ),
    # Case 1: Simple case with negative correlation
    (
        {
            "x": np.array([1.0, 2.0, 3.0, 4.0]),
            "yhat": np.array([-1.0, -2.0, -3.0, -4.0]),
            "t_idx": np.array([0, 1, 2, 3])
        },
        -1.0 # expected beta_hat
    ),
    # Case 2: Zero denominator (should raise ValueError)
    (
        {
            "x": np.array([0.0, 0.0, 0.0, 0.0]),
            "yhat": np.array([1.0, 2.0, 3.0, 4.0]),
            "t_idx": np.array([0, 1, 2, 3])
        },
        ValueError # raises ValueError due to zero denominator
    ),
    # Case 3: Mismatched lengths (should raise ValueError)
    (
        {
            "x": np.array([1.0, 2.0, 3.0]),
            "yhat": np.array([1.0, 2.0, 3.0, 4.0]),
            "t_idx": np.array([0, 1, 2])
        },
        ValueError # raises ValueError due to mismatched lengths
    )
]
@pytest.mark.parametrize("input_run,expected_output", beta_hat_testdata)
def test_beta_hat(input_run, expected_output):
    """
    Tests the `beta_hat` function to ensure it correctly computes 
    the model-implied AR(1) beta.
    """
    # Check if we expect a ValueError class or instance
    if expected_output == ValueError or isinstance(expected_output, ValueError):
        with pytest.raises(ValueError):
            beta_hat(input_run)
    else:
        assert beta_hat(input_run) == expected_output


##################### Tests for average_power_spectrum function ###################
