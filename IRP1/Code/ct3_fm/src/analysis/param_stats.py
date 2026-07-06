"""
This file contains functions used to compute the parameter statistics for each generator.

""" 

from typing import Dict, Any
import numpy as np
from scipy import signal
from statsmodels.tsa.ar_model import AutoReg

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

    Measures the autoregressive parameter for the forecasted trajectory.
    Requires:

      - run["yhat"]  : forecasted trajectory

    Args:
        run (Run): An instance of the Run class containing the trajectory data.
    
    Returns:
        float: The model-implied AR(1) beta.
    """
    yhat = np.asarray(run["yhat"], dtype=float)

    # check if the lengths of x, yhat, and t_idx are compatible
    if len(yhat) < 1 :
        raise ValueError("Input arrays must not be empty.")
    
    # Fit an AR model with a lag of 1
# trend='c' includes a constant/intercept if your data isn't mean-centered
    model = AutoReg(yhat, lags=1, trend='c').fit()

    # Extract the beta coefficient for the first lag
    # index 0 is the constant (intercept), index 1 is the lag-1 coefficient
    beta = float(model.params[1]) 

    return beta

def dominant_frequency(run: Run) ->  tuple[np.ndarray, np.ndarray]:
    """
    Computes the dominant frequency of the trajectory for a given run by computing the power spectrum and finding the frequency with the highest power.


    Requires:
      - run["yhat"]  : forecasted trajectory

    Args:
        run (Run): An instance of the Run class containing the trajectory data.
    
    Returns:
        float: The dominant frequency of the trajectory.
    """
    
    yhat = np.asarray(run["yhat"], dtype=float)
    t_idx = np.asarray(run["t_idx"], dtype=int)

    # check if the lengths of yhat and t_idx are compatible
    if len(yhat) < 1 or len(t_idx) < 1:
        raise ValueError("Input arrays must not be empty.")
    if not (len(yhat) == len(t_idx)):
        raise ValueError("Input arrays must have the same length.")

 
    frequencies, power = signal.welch(yhat, fs=1)
    dominant_freq = frequencies[np.argmax(power)]

    
    return dominant_freq

# Set up parameter statistics registry

PARAM_STATS_REGISTRY = {
    "prob_positive": prob_positive,
    "beta_hat": beta_hat,
    "dominant_frequency": dominant_frequency    
}
