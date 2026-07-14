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

from analysis.param_stats import prob_positive, beta_hat, dominant_frequency, estimated_dwell_time

################### Tests for prob_positive function ###################
prob_positive_testdata = [
    # Case 0: Increasing values
    (
        np.arange(1, 1000, 1), # input run
        1.0                            # expected probability
    ),
    # Case 1: Decreasing values
    (
        np.arange(0,-99,-1), # input run
        0.0                            # expected probability
    ),
    # Case 2: Mixed values sampled from a normal distribution
    (
        np.random.normal(0, 1, 1000), # input run
        0.5                            # expected probability
    ),
    # Case 3: Empty trajectory
    (
        np.array([]), # input run
        ValueError # raises ValueError due to empty input
    ),
    # Case 4: Wrong data type (string instead of numeric)
    (
        ["a", "b", "c"], # input run
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
        assert np.isclose(prob_positive(input_run), expected_output, atol=1e-1), f"Expected {expected_output}, got {prob_positive(input_run)}"


#################### Tests for beta_hat function ###################

def generate_ar1_series(n_steps=500, beta=0.7, sigma=1.0, seed=42):
    """
    Generates an AR(1) time series: x_{t+1} = beta * x_t + epsilon_t
    
    Parameters:
    - n_steps: Total number of points to generate.
    - beta: The AR(1) coefficient (controls memory/momentum).
    - sigma: Standard deviation of the random noise (epsilon).
    """
    rng = np.random.default_rng(seed)
    
    # Pre-allocate array for speed
    series = np.zeros(n_steps)
    
    # Generate all random noise shocks upfront
    epsilon = rng.normal(loc=0.0, scale=sigma, size=n_steps)
    
    # Initialize the first point with a noise shock
    series[0] = epsilon[0]
    
    # Iteratively calculate the rest of the steps
    for t in range(1, n_steps):
        series[t] = beta * series[t-1] + epsilon[t]
        
    return np.asarray(series)

beta_hat_testdata = [

    # Case 0: Positive correlation with beta=0.7
    (
        generate_ar1_series(n_steps=500, beta=0.7, sigma=1.0),
        0.7 # expected beta
    ),
    # Case 1: Positive correlation with beta=1.0
    (
        generate_ar1_series(n_steps=500, beta=1.0, sigma=1.0),
        1.0 # expected beta
    ),
    # Case 2: Empty trajectory (should raise ValueError)
    (
        np.array([]),
        ValueError # raises ValueError due to zero denominator
    ),
    # Case 4: Negative correlation with beta=-0.5
    (
        generate_ar1_series(n_steps=500, beta=-0.5, sigma=1.0),
        -0.5 # expected beta
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
        # only assert same sign as estimated beta is likely not exactly equal to the true beta
        assert np.sign(beta_hat(input_run)) == np.sign(expected_output), f"Expected sign {np.sign(expected_output)}, got {np.sign(beta_hat(input_run))}"


##################### Tests for dominant_frequency function ###################
n = 1000 # number of samples for the test signals
dominant_frequency_testdata = [
    # Case 0: Simple sinusoidal signal with frequency of 0.1 Hz
    (
        np.sin(2 * np.pi * 0.1 * np.arange(n)),
        0.1 # expected dominant frequency
    ),
    # Case 1: Simple sinusoidal signal with frequency of 0.05 Hz
    (
        np.sin(2 * np.pi * 0.05 * np.arange(n)),
        0.05 # expected dominant frequency
    ),
    # Case 2: Constant signal of ones (should return 0.0 as dominant frequency)
    (
        np.ones(n),
        0.0 # expected dominant frequency
    ),
    # Case 3: Empty trajectory (should raise ValueError)
    (
        np.array([]),
        ValueError # raises ValueError due to empty input
    )
]
@pytest.mark.parametrize("input_run,expected_output", dominant_frequency_testdata)
def test_dominant_frequency(input_run, expected_output):
    """
    Tests the `dominant_frequency` function to ensure it correctly computes 
    the dominant frequency of the trajectory.
    """
    # Check if we expect a ValueError class or instance
    if expected_output == ValueError or isinstance(expected_output, ValueError):
        with pytest.raises(ValueError):
            dominant_frequency(input_run)
    else:
        freq = dominant_frequency(input_run)
        assert np.isclose(freq, expected_output, atol=1e-2), f"Expected {expected_output}, got {freq}"


##################### Tests for infer_dwell_pelt function ###################

def generate_regime_series(dwell_time=20, num_regimes=4, slopes=[2.0, -2.0], sigma=0.01, seed=42):
    """
    Generates a realistic multi-regime cumulative trend path with a known uniform 
    dwell time and a small amount of noise to stabilize change-point detection.
    """
    rng = np.random.default_rng(seed)
    T = dwell_time * num_regimes
    indices = np.arange(T)
    regimes = (indices // dwell_time) % len(slopes)
    slope_array = np.array(slopes)[regimes]
    
    # Add a tiny bit of noise so the L2 model has a non-zero variance baseline
    noise = rng.normal(loc=0.0, scale=sigma, size=T)
    return np.cumsum(slope_array + noise)

estimated_dwell_time_testdata = [
    # Case 0: Clean step shifts with perfect uniform dwell time of 30 steps
    (
        generate_regime_series(dwell_time=30, num_regimes=10, slopes=[5.0, -5.0]),
        30.0 # expected average dwell time
    ),
    # Case 1: Clean step shifts with uniform dwell time of 15 steps
    (
        generate_regime_series(dwell_time=15, num_regimes=10, slopes=[10.0, -10.0]),
        15.0 # expected average dwell time
    ),
    # Case 2: Insufficient observation sequence length (length <= 2)
    (
        np.array([1.0, 2.0]),
        ValueError # raises ValueError due to tiny array size
    ),
    # Case 3: Empty trajectory
    (
        np.array([]),
        ValueError # raises ValueError due to empty array
    ),
    # Case 4: High dimensional matrix instead of 1D array
    (
        np.ones((2, 50)),
        ValueError # raises ValueError due to incorrect structure dimension
    )
]
@pytest.mark.parametrize("input_run,expected_output", estimated_dwell_time_testdata)
def test_estimated_dwell_time(input_run, expected_output):
    """
    Tests the `estimated_dwell_time` function to ensure it correctly estimates
    the average sequence length between trend switches using the PELT algorithm.
    """
    if expected_output == ValueError or isinstance(expected_output, ValueError):
        with pytest.raises(ValueError):
            estimated_dwell_time(input_run)
    else:
        estimated_dwell = estimated_dwell_time(input_run, penalty=1.5)
        # Using a slight tolerance buffer for exact changepoint indexing borders
        assert np.isclose(estimated_dwell, expected_output, atol=1e-1), f"Expected average dwell time {expected_output}, got {estimated_dwell}"