"""
This file contains functions used to compute the parameter statistics for each generator.
""" 

import numpy as np
from scipy import signal
from statsmodels.tsa.ar_model import AutoReg



def prob_positive(trajectory: np.array) -> float:
    """
    Computes the probability of positive changes between time step values in the trajectory.

    Args:
      - trajectory: sequence of time series values (1D array)

    Returns:
      - float: The probability of positive change in values in the trajectory.

    """
       
    if trajectory is None or len(trajectory) == 0:
        raise ValueError("Invalid input data type for 'trajectory' or empty array")

    try:
        # Squeeze to eliminate potential single-dimensional axes like (1, 200)
        trajectory = np.asarray(trajectory, dtype=float).squeeze()
        prob_pos = float(np.mean(trajectory[1:] > trajectory[:-1])) # Probability of positive change
        
        if np.isnan(prob_pos):
            raise ValueError("Invalid input data type for 'trajectory'")

    except (TypeError, ValueError):
        raise ValueError("Invalid input data type for 'trajectory'")

    return prob_pos


def beta_hat(trajectory: np.array) -> float:
    """
    Computes the model-implied AR(1) beta for a given run.
    
    Args:
      - trajectory  : forecasted trajectory

    Returns:
        - float: The estimated AR(1) beta coefficient.  
    """
    # Squeeze the array to force a 1D structure (e.g., shape (200,))
    # If the shape is (1, 200), statsmodels thinks nobs = 1, triggering 'maxlag should be < nobs'
    trajectory = np.asarray(trajectory, dtype=float).squeeze()

    if trajectory.ndim > 1:
        raise ValueError(f"Expected 1D trajectory array, got shape {trajectory.shape}")
        
    if len(trajectory) <= 1:
        raise ValueError("Input arrays must contain more than 1 observation to fit AR(1).")
    
    # Fit an AR model with a lag of 1
    model = AutoReg(trajectory, lags=1, trend='c').fit()

    # Extract the beta coefficient for the first lag
    beta = float(model.params[1]) 

    return beta


def dominant_frequency(trajectory: np.array) -> float:
    """
    Computes the dominant frequency of the trajectory.

    Args:
      - trajectory  : forecasted trajectory

    Returns:
        - float: The dominant frequency of the trajectory.
    """
    # Squeeze array to 1D
    tra = np.asarray(trajectory, dtype=float).squeeze()

    if len(trajectory) < 1:
        raise ValueError("Input arrays must not be empty.")

    # Compute power spectrum density using Welch's method
    frequencies, power = signal.welch(trajectory, fs=1)
    dominant_freq = float(frequencies[np.argmax(power)])

    return dominant_freq


# Set up parameter statistics registry
PARAM_STATS_REGISTRY = {
    "prob_positive": prob_positive,
    "beta_hat": beta_hat,
    "dominant_frequency": dominant_frequency    
}
