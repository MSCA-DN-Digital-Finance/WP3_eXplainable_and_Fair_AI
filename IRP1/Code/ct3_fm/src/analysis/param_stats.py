"""
This file contains functions used to compute the parameter statistics for each generator.
""" 

from typing import Any
import numpy as np
from scipy import signal
from statsmodels.tsa.ar_model import AutoReg
import ruptures as rpt


def _ensure_1d_trajectory(trajectory: Any) -> np.ndarray:
    """
    Safely converts input to a 1D float NumPy array.
    Prevents 1-element inputs from collapsing into 0D arrays after squeezing.
    """
    if trajectory is None:
        raise ValueError("Trajectory cannot be None")
        
    arr = np.asarray(trajectory, dtype=float).squeeze()
    
    # If a 1-element array got squeezed down to a 0D scalar, restore it to 1D
    if arr.ndim == 0:
        arr = np.atleast_1d(arr)
        
    if arr.ndim > 1:
        raise ValueError(f"Expected 1D trajectory array, got shape {arr.shape}")
        
    return arr


def prob_positive(trajectory: np.ndarray) -> float:
    """
    Computes the probability of positive changes between time step values in the trajectory.

    Args:
      - trajectory: sequence of time series values (1D array)

    Returns:
      - float: The probability of positive change in values in the trajectory.
    """
    try:
        arr = _ensure_1d_trajectory(trajectory)
    except (TypeError, ValueError) as e:
        raise ValueError("Invalid input data type for 'trajectory' or empty array") from e

    if len(arr) <= 1:
        raise ValueError("Invalid input data type for 'trajectory' or empty array")

    prob_pos = float(np.mean(arr[1:] > arr[:-1])) # Probability of positive change
    
    if np.isnan(prob_pos):
        raise ValueError("Invalid input data type for 'trajectory'")

    return prob_pos


def beta_hat(trajectory: np.ndarray) -> float:
    """
    Computes the model-implied AR(1) beta for a given run.
    
    Args:
      - trajectory  : forecasted trajectory

    Returns:
        - float: The estimated AR(1) beta coefficient.  
    """
    arr = _ensure_1d_trajectory(trajectory)
        
    if len(arr) <= 1:
        raise ValueError("Input arrays must contain more than 1 observation to fit AR(1).")
    
    # Fit an AR model with a lag of 1
    model = AutoReg(arr, lags=1, trend='c').fit()

    # Extract the beta coefficient for the first lag
    beta = float(model.params[1]) 

    return beta


def dominant_frequency(trajectory: np.ndarray) -> float:
    """
    Computes the dominant frequency of the trajectory.

    Args:
      - trajectory  : forecasted trajectory

    Returns:
        - float: The dominant frequency of the trajectory.
    """
    arr = _ensure_1d_trajectory(trajectory)

    if len(arr) <= 1:
        raise ValueError("Input arrays must contain more than 1 observation to calculate frequency.")

    # Compute power spectrum density using Welch's method
    frequencies, power = signal.welch(arr, fs=1)
    dominant_freq = float(frequencies[np.argmax(power)])

    return dominant_freq


def estimated_dwell_time(trajectory: np.ndarray, penalty: float = 1.5) -> float:
    """
    Detects change points in a noisy trend using PELT and 
    calculates the average dwell time.

    Args:
      - trajectory : sequence of time series values (1D array)
      - penalty    : PELT penalty complexity parameter (sensitivity)

    Returns:
      - float: The average dwell time.
    """
    arr = _ensure_1d_trajectory(trajectory)

    if len(arr) <= 2:
        raise ValueError("Input arrays must contain more than 2 observations to calculate differences and dwell times.")
    
    # 1. Convert trend to differences (slopes)
    # This turns 'slope changes' into 'mean changes'
    signal_diff = np.diff(arr)
    
    # 2. Configure PELT
    # 'l2' (Least Squares) is best for shifts in the mean
    algo = rpt.Pelt(model="l2", jump=1).fit(signal_diff)
    
    # 3. Predict change points
    # The 'pen' value is the sensitivity. 
    result = algo.predict(pen=penalty)
    
    # 4. Calculate Dwell Times
    change_points = [0] + result
    dwell_times = np.diff(change_points)
    
    avg_dwell = float(np.mean(dwell_times))
        
    return avg_dwell


def estimated_threshold(trajectory: np.ndarray) -> float:
    """
    Finds the reset points (peaks) in an Integrate-and-Fire series and returns 
    their average height. Falls back to global max if no drops are found.

    Args:
      - trajectory: sequence of time series values (1D array)

    Returns:
      - float: The estimated threshold parameter.
    """
    arr = _ensure_1d_trajectory(trajectory)
    
    if len(arr) == 0:
        raise ValueError("Input arrays must not be empty.")
    if len(arr) == 1:
        return float(arr[0])

    # Find indices where the value drops significantly (the reset)
    diffs = np.diff(arr)
    
    # A large negative jump indicates a reset
    peak_indices = np.where(diffs < -0.5 * np.max(arr))[0]
    
    if len(peak_indices) == 0:
        return float(np.max(arr)) # Fallback to global max
        
    # The peak is the value right before the jump
    peaks = arr[peak_indices]
    return float(np.mean(peaks))


# Set up parameter statistics registry
PARAM_STATS_REGISTRY = {
    "prob_positive": prob_positive,
    "beta_hat": beta_hat,
    "dominant_frequency": dominant_frequency,
    "estimated_dwell_time": estimated_dwell_time,
    "estimated_threshold": estimated_threshold,
}