import numpy as np
import pandas as pd

from typing import Dict, Any, Tuple


def chronos_inference(
    model: Any,
    input_df: pd.DataFrame,
    output_length: int,
    quantiles: list = [0.5]
    ) -> pd.DataFrame:

    pred_df = model.predict_df(input_df, prediction_length=output_length, quantile_levels=quantiles)

    # return predictions as (sample, time steps, dimensions) numpy array

    pred_array = np.array()

    return pred_df

