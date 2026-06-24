"""
This file contains functions used to compute the parameter statistics for each generator.

""" 

from pdb import run
from typing import Dict, Any
import numpy as np


Run = Dict[str, Any]


def prob_positive(run:Run) -> float:
    """
    Computes the probability of positive values in the trajectory for a given run.

        Requires:
      - run["yhat"]  : 1-step-ahead level forecasts

    Args:
        run (Run): An instance of the Run class containing the trajectory data.
    
    Returns:
        float: The probability of positive values in the trajectory.
    """
    # retrieve the trajectory from the run dictionary
    yhat = run.get("yhat")
    
    # Check if yhat is None or empty to prevent numpy NaN warnings
    if yhat is None or len(yhat) == 0:
        raise ValueError("Invalid input data type for 'yhat' or empty array")

    try:
        prob_pos = float(np.mean(np.asarray(yhat) > 0.0))
        
        # Catch edge case where numpy somehow still outputs nan
        if np.isnan(prob_pos):
            raise ValueError("Invalid input data type for 'yhat'")

    except (TypeError, ValueError):
        raise ValueError("Invalid input data type for 'yhat'")

    return prob_pos


def beta_hat(run: Run, eps: float = 1e-12) -> float:
    """
    Computes the model-implied AR(1) beta for a given run.

    Model-implied AR(1) beta:
        beta_hat = sum(x_t * yhat_{t+1}) / sum(x_t^2)

    Measures how strongly the model's forecast depends on the current state.
    Requires:
      - run["x"]     : full level trajectory
      - run["yhat"]  : 1-step-ahead level forecasts
      - run["t_idx"] : indices such that yhat[i] predicts x[t_idx[i] + 1]

    Args:
        run (Run): An instance of the Run class containing the trajectory data.
        eps (float): A small value to prevent division by zero.
    
    Returns:
        float: The model-implied AR(1) beta.
    """
    x = np.asarray(run["x"], dtype=float)
    yhat = np.asarray(run["yhat"], dtype=float)
    t_idx = np.asarray(run["t_idx"], dtype=int)

    # check if the lengths of x, yhat, and t_idx are compatible
    if len(x) < 1 or len(yhat) < 1 or len(t_idx) < 1:
        raise ValueError("Input arrays must not be empty.")
    
    # check if the lengths of x, yhat, and t_idx are the same
    if not (len(x) == len(yhat) == len(t_idx)):
        raise ValueError("Input arrays must have the same length.")

    # x_t aligned with yhat
    x_t = x[t_idx]

    denom = float(np.dot(x_t, x_t))
    if denom < eps:
        raise ValueError("Denominator is too small, potential division by zero.")

    return float(np.dot(x_t, yhat) / denom)

def average_power_spectrum(run: Run) ->  tuple[np.ndarray, np.ndarray]:
    """
    Computes the average power spectrum of the trajectory for a given run by computing power spectra per window and then averaging them.


    Requires:
      - ?

    Args:
        run (Run): An instance of the Run class containing the trajectory data.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: The average power spectrum of the trajectory.
    """
    pass
