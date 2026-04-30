import numpy as np
from typing import Tuple

def build_samples(
    traj: np.ndarray, 
    input_length: int, 
    gap: int, 
    output_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs input-output pairs for time series prediction.
    
    Parameters:
    - traj: List of NumPy arrays, each representing a trajectory.
    - input_length: Number of past time steps to include in the input (x).
    - gap: Number of time steps to skip between the input and output.
    - output_length: Number of future time steps to predict (length of y).
    
    Returns:
    - x: NumPy array of shape (n_samples, input_length, dim) containing the input sequences.
    - y: NumPy array of shape (n_samples, output_length, dim) containing the target sequences.
    
    Note:
    The function assumes that the trajectory has enough data points to construct the sample
    based on the provided parameters. It does not perform bounds checking.
    """
    if traj.ndim == 1:
            traj = traj[:, np.newaxis]
        
    T, D = traj.shape
    num_samples = T - input_length - gap - output_length + 1
    
    if num_samples <= 0:
        raise ValueError("Trajectory too short for given parameters.")

    # Pre-allocate 3D arrays (N, Time, Dim)
    x = np.zeros((num_samples, input_length, D), dtype=traj.dtype)
    y = np.zeros((num_samples, output_length, D), dtype=traj.dtype)

    for i in range(num_samples):
        x[i] = traj[i : i + input_length]
        y[i] = traj[i + input_length + gap : i + input_length + gap + output_length]

    return x, y