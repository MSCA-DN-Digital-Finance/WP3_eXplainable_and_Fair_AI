import pandas as pd
import numpy as np
from typing import Any, Dict, Callable

from .adapters import chronos_input_adapter, chronos_output_adapter, timesfm_input_adapter, timesfm_output_adapter



############################ Model Loading #####################################

def load_chronos_model(model_id: str = "amazon/chronos-2") -> Any:
    """
    Loads the Chronos-2 forecasting pipeline optimized for CPU execution.

    Given that this runs on a ThinkPad, we explicitly set the device to 'cpu'.
    We also utilize float32 precision, as most laptop CPUs do not provide
    significant speedups for half-precision (fp16) compared to GPUs.

    Args:
        model_id (str): The HuggingFace model hub ID. 
                        Defaults to "amazon/chronos-2".

    Returns:
        Chronos2Pipeline: The loaded and initialized Chronos pipeline.
    """
    import torch
    from chronos import BaseChronosPipeline, Chronos2Pipeline
    
    # 1. Explicitly define the device as CPU.
    # On a laptop, 'cuda' is likely unavailable, and 'auto' can sometimes 
    # default to a ghostly GPU reference that causes errors.
    device = "cpu"

    # 2. Load the pipeline from the pretrained source.
    # We use torch_dtype=torch.float32 for maximum compatibility on CPU.
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.float32,
    )

    # 3. Optimization: Turn off gradient calculation globally for inference.
    # This reduces memory overhead on your ThinkPad's RAM.
    torch.set_grad_enabled(False)

    return pipeline



def load_timesfm_model(model_id: str = "google/timesfm-2.5-200m-pytorch") -> Any:
    """
    Loads and compiles the TimesFM model for CPU inference.

    This function optimizes the PyTorch backend for Intel/AMD CPUs and 
    configures the forecasting head with robust constraints (positivity, 
    quantile crossing fix, etc.).

    Args:
        model_id (str): The HuggingFace model hub ID.
                        Defaults to the 2.5-200m version.

    Returns:
        model: The compiled TimesFM model ready for .forecast() calls.
    """
    import timesfm
    import torch
    
    # 1. Optimize Float32 performance for CPU.
    # "high" or "medium" allows the use of TensorFloat-32 (on supported CPUs) 
    # or optimized BLAS libraries, providing a boost over "highest".
    torch.set_float32_matmul_precision("high")

    # 2. Load the model weights.
    # TimesFM 2.5 200M is a medium-sized model; ensure your ThinkPad has 
    # at least 16GB of RAM for a comfortable experience.
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)

    # 3. Configure and Compile the model.
    # Compilation optimizes the execution graph for your specific hardware.
    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,              # Max history the model can look at
            max_horizon=256,               # Max future steps it can predict
            normalize_inputs=True,         # Recommended for stability
            use_continuous_quantile_head=True,
            force_flip_invariance=True,    # Ensures [1, 2, 3] yields same logic as [3, 2, 1]
            infer_is_positive=True,        # Useful for most physical/economic metrics
            fix_quantile_crossing=True,    # Ensures 90th quantile is always > 10th
        )
    )

    # 4. Explicitly move to CPU and set to eval mode.
    # Though .from_pretrained usually handles this, we be explicit for the registry.
    model.to("cpu")
    torch.set_grad_enabled(False)

    return model


########################### Inference Adapters #####################################

def chronos_inference(
    model: Any,
    input_df: pd.DataFrame,
    output_length: int,
    quantiles: list = [0.5]
) -> pd.DataFrame:
    """
    Executes a forecast using the Chronos model and returns a long-format DataFrame.

    This function acts as the core inference bridge for Chronos, taking a 
    standardized pandas DataFrame and producing future predictions for all 
    unique item IDs present in the input.

    Args:
        model (Any): The loaded Chronos pipeline or model object.
        input_df (pd.DataFrame): Input data containing 'item_id', 'timestamp', 
            and 'target' columns.
        output_length (int): The number of future time steps to forecast 
            (the horizon).
        quantiles (list): The specific quantile levels to calculate. 
            Defaults to [0.5] (the median point forecast).

    Returns:
        pd.DataFrame: A "long" format DataFrame containing the future 
            predictions, timestamps, and requested quantiles for each item_id.
    """

    # 1. Generate predictions using the model's internal predict_df method.
    # Chronos handles the context padding and frequency detection internally.
    pred_df = model.predict_df(
        input_df, 
        prediction_length=output_length, 
        quantile_levels=quantiles
    )

    # Note: Chronos returns a DataFrame where column names match the 
    # quantile strings (e.g., "0.5"). This DataFrame is already indexed 
    # by item_id, making it ready for post-processing or stacking.
    return pred_df




def timesfm_inference(
    model: Any,
    input_list: list[np.ndarray],
    output_length: int
) -> np.ndarray:
    """
    Executes a forecast using the TimesFM model and returns the point forecasts.

    This function serves as the inference bridge in the model registry. It 
    handles the specific output format of TimesFM (a tuple of point and 
    quantile forecasts) and extracts the primary prediction array.

    Args:
        model (Any): The loaded TimesFM model object.
        input_list (list[np.ndarray]): A list of 1D NumPy arrays, where each 
            array represents a single univariate time series.
        output_length (int): The number of future time steps to forecast 
            (horizon).

    Returns:
        np.ndarray: A 2D NumPy array of shape (N_sequences, output_length) 
            containing the point forecasts.
    """

    # 1. Generate the forecast.
    # TimesFM returns a tuple: (point_forecast, quantile_forecast)
    # point_forecast shape: (N, horizon)
    # quantile_forecast shape: (N, horizon, n_quantiles)
    point_forecast, _ = model.forecast(
        inputs=input_list,
        horizon=output_length
    )

    # 2. Return only the point forecast array.
    return point_forecast



######################## Model Registry ##########################


MODEL_REGISTRY: Dict[str, Dict[str, Callable]] = {
    "timesfm": {
        "loader": load_timesfm_model,
        "input_adapter": timesfm_input_adapter,
        "inference_fn": timesfm_inference,
        "output_adapter": timesfm_output_adapter,
    },
    "chronos": {
        "loader": load_chronos_model,
        "input_adapter": chronos_input_adapter,
        "inference_fn": chronos_inference,
        "output_adapter": chronos_output_adapter,
    }
}