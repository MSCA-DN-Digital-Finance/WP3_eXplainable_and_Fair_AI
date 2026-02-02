from __future__ import annotations

from pathlib import Path
from pyexpat import model
import numpy as np
import pandas as pd

from typing import Dict, Any, Tuple





def load_npz_series(run_dir: Path, key: str = "x") -> np.ndarray:
    data = np.load(run_dir / "trajectory.npz")
    if key not in data:
        raise KeyError(f"{key} not in trajectory.npz. Keys: {list(data.keys())}")
    return np.asarray(data[key], dtype=np.float32)


def make_target(x: np.ndarray, target_kind: str) -> np.ndarray:
    """
    target_kind:
      - "x": levels (same length as x)
      - "diff": first difference (length len(x)-1)
    """
    if target_kind == "x":
        return x
    if target_kind == "diff":
        return np.diff(x)
    raise ValueError("target_kind must be 'x' or 'diff'")


### Chronos prediction functions ###


def chronos_rolling_1step_predict(
    pipeline,
    y: np.ndarray,
    *,
    run_id: str,
    window_length: int = 50,
    start_date: str = "2020-01-01",
    freq: str = "D",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - yhat: shape (N_windows,) one-step median forecasts
      - t_idx: shape (N_windows,) integer indices in the target y such that
              each yhat[i] predicts y[t_idx[i] + 1]
    """
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if window_length < 2:
        raise ValueError("window_length must be >= 2")
    if len(y) <= window_length:
        raise ValueError("len(y) must be > window_length for 1-step forecasts")

    timestamps = pd.date_range(start=start_date, periods=len(y), freq=freq)

    yhat = []
    t_idx = []

    # window covers indices [start_idx, end_idx-1], forecast next step end_idx
    for start_idx in range(0, len(y) - window_length):
        end_idx = start_idx + window_length
        window_y = y[start_idx:end_idx]

        window_df = pd.DataFrame(
            {
                "id": run_id,  # constant id for this run
                "timestamp": timestamps[start_idx:end_idx],
                "target": window_y,
            }
        )

        pred_df = pipeline.predict_df(
            df=window_df,
            prediction_length=1,
            quantile_levels=[0.5],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        # Chronos returns a DF; extract the median prediction.
        # Column name can vary by implementation; try common ones.
        if "0.5" in pred_df.columns:
            median = float(pred_df["0.5"].iloc[0])
        elif "q0.5" in pred_df.columns:
            median = float(pred_df["q0.5"].iloc[0])
        elif "target" in pred_df.columns:
            # some APIs return forecast in "target"
            median = float(pred_df["target"].iloc[0])
        else:
            raise KeyError(f"Unknown prediction column. Got columns: {list(pred_df.columns)}")

        yhat.append(median)
        t_idx.append(end_idx - 1)  # last index in window; predicts next step

    return np.asarray(yhat, dtype=np.float32), np.asarray(t_idx, dtype=np.int32)




def iter_runs(gen_root: Path):
    for run_dir in sorted(gen_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if (run_dir / "trajectory.npz").exists() and (run_dir / "meta.json").exists():
            yield run_dir





def chronos_ar1_rolling_1step_predict_levels(
    pipeline,
    x: np.ndarray,
    *,
    run_id: str,
    window_length: int,
    start_date: str = "2020-01-01",
    freq: str = "D",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronos rolling 1-step-ahead prediction for AR(1) *levels*.

    Inputs:
      x: level trajectory of shape (T,)
      window_length: context length used for each forecast

    Outputs:
      yhat: (N,) median 1-step forecasts for x_{t+1}
      t_idx: (N,) indices t such that yhat[i] predicts x[t_idx[i] + 1]
      x_last: (N,) last observed value x_t used in the context window (useful for diagnostics)
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if window_length < 2:
        raise ValueError("window_length must be >= 2")
    if len(x) <= window_length:
        raise ValueError("len(x) must be > window_length")

    timestamps = pd.date_range(start=start_date, periods=len(x), freq=freq)

    yhat = []
    t_idx = []
    x_last = []

    # window = x[start:end], predict x[end]
    for start in range(0, len(x) - window_length):
        end = start + window_length

        window_df = pd.DataFrame(
            {
                "id": run_id,
                "timestamp": timestamps[start:end],
                "target": x[start:end],
            }
        )

        pred_df = pipeline.predict_df(
            df=window_df,
            prediction_length=1,
            quantile_levels=[0.5],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        # Extract median forecast (API column naming can vary)
        for col in ("0.5", "q0.5", "target"):
            if col in pred_df.columns:
                median = float(pred_df[col].iloc[0])
                break
        else:
            raise KeyError(f"Unknown prediction column. Got: {list(pred_df.columns)}")

        yhat.append(median)
        t_idx.append(end - 1)          # last index in context window
        x_last.append(float(x[end - 1]))

    return (
        np.asarray(yhat, dtype=np.float32),
        np.asarray(t_idx, dtype=np.int32),
        np.asarray(x_last, dtype=np.float32),
    )





def chronos_harmonic_multistep_forecast(
    pipeline,
    x: np.ndarray,
    *,
    run_id: str,
    window_length: int = 100,
    prediction_length: int = 200,
    step_size: int = 1,
    quantile_levels: Tuple[float, ...] = (0.5,),
    start_date: str = "2020-01-01",
    freq: str = "D",
) -> Dict[str, Any]:
    """
    Sliding-window multi-step forecasting on a harmonic oscillator level series.

    Returns a compact dict of arrays:
      - yhat_q: (N_windows, prediction_length, Q) quantile forecasts
      - window_start: (N_windows,)
      - window_end: (N_windows,)  (index of last observed point in the window)
      - x_last: (N_windows,)
      - quantile_levels: (Q,)
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if len(x) <= window_length:
        raise ValueError("len(x) must be > window_length")
    if prediction_length <= 0:
        raise ValueError("prediction_length must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    q_levels = tuple(float(q) for q in quantile_levels)
    q_cols = [str(q) for q in q_levels]  # Chronos uses strings like "0.5" often

    timestamps = pd.date_range(start=start_date, periods=len(x), freq=freq)

    yhat_all = []
    w_start = []
    w_end = []
    x_last = []

    for start_idx in range(0, len(x) - window_length, step_size):
        end_idx = start_idx + window_length  # window includes [start_idx, end_idx-1]
        window = x[start_idx:end_idx]

        window_df = pd.DataFrame(
            {
                "id": run_id,
                "timestamp": timestamps[start_idx:end_idx],
                "target": window,
            }
        )

        pred_df = pipeline.predict_df(
            df=window_df,
            prediction_length=prediction_length,
            quantile_levels=list(q_levels),
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        # Extract quantile columns robustly
        # (Chronos may return "0.5" or "q0.5" depending on implementation)
        cols_found = []
        for q in q_levels:
            c1 = str(q)
            c2 = f"q{q}"
            if c1 in pred_df.columns:
                cols_found.append(c1)
            elif c2 in pred_df.columns:
                cols_found.append(c2)
            else:
                raise KeyError(f"Missing quantile column for {q}. Got: {list(pred_df.columns)}")

        # pred_df is long with prediction_length rows; stack into (H, Q)
        arr_hq = pred_df[cols_found].to_numpy(dtype=np.float32)  # (H, Q)

        yhat_all.append(arr_hq)
        w_start.append(start_idx)
        w_end.append(end_idx - 1)
        x_last.append(float(window[-1]))

    yhat_q = np.stack(yhat_all, axis=0)  # (N_windows, H, Q)

    return {
        "yhat_q": yhat_q,
        "window_start": np.asarray(w_start, dtype=np.int32),
        "window_end": np.asarray(w_end, dtype=np.int32),
        "x_last": np.asarray(x_last, dtype=np.float32),
        "quantile_levels": np.asarray(q_levels, dtype=np.float32),
    }



### TimesFM prediction functions ###


def timesfm_rolling_1step_predict(
    model,
    y: np.ndarray,
    *,
    window_length: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    TimesFM analogue of chronos_rolling_1step_predict.

    Returns:
      - yhat: shape (N_windows,) one-step point forecasts (TimesFM point forecast)
      - t_idx: shape (N_windows,) integer indices in the target y such that
              each yhat[i] predicts y[t_idx[i] + 1]

    Notes:
      - TimesFM `forecast` takes raw arrays; no timestamps / ids needed.
      - We keep the same alignment convention as your Chronos helper.
    """
    y = np.asarray(y, dtype=np.float32)

    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if window_length < 2:
        raise ValueError("window_length must be >= 2")
    if len(y) <= window_length:
        raise ValueError("len(y) must be > window_length for 1-step forecasts")

    yhat = []
    t_idx = []

    # window covers indices [start_idx, end_idx-1], forecast next step end_idx
    for start_idx in range(0, len(y) - window_length):
        end_idx = start_idx + window_length
        window_y = y[start_idx:end_idx]

        inp = np.asarray(window_y, dtype=np.float32)[None, :]   # (1, context)
        point_forecast, _quantile_forecast = model.forecast(horizon=1, inputs=inp)

        pf = np.asarray(point_forecast, dtype=np.float32)
        if not np.isfinite(pf).all():
            raise RuntimeError(
                f"TimesFM produced non-finite forecast. "
                f"window stats: min={float(np.min(inp))}, mean={float(np.mean(inp))}, max={float(np.max(inp))}, "
                f"std={float(np.std(inp))}"
            )
        yhat.append(float(pf[0, 0]))
        t_idx.append(end_idx - 1)

    return np.asarray(yhat, dtype=np.float32), np.asarray(t_idx, dtype=np.int32)



def timesfm_harmonic_multistep_forecast(
    model,
    x: np.ndarray,
    *,
    window_length: int = 100,
    prediction_length: int = 200,
    step_size: int = 1,
) -> Dict[str, Any]:
    """
    TimesFM analogue of chronos_harmonic_multistep_forecast.

    Sliding-window multi-step forecasting on a harmonic oscillator level series.

    Returns a compact dict of arrays:
      - yhat: (N_windows, prediction_length) point forecasts
      - window_start: (N_windows,)
      - window_end: (N_windows,)  (index of last observed point in the window)
      - x_last: (N_windows,)

    Notes
    -----
    - TimesFM requires `model.compile(ForecastConfig(...))` *before* calling forecast().
    - TimesFM `forecast(horizon=H, inputs=[window])` returns:
        point_forecast: (1, H)
        quantile_forecast: (1, H, K)  (optional depending on config)
      Here we store only point forecasts for simplicity/stability.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if len(x) <= window_length:
        raise ValueError("len(x) must be > window_length")
    if prediction_length <= 0:
        raise ValueError("prediction_length must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    yhat_all = []
    w_start = []
    w_end = []
    x_last = []

    for start_idx in range(0, len(x) - window_length, step_size):
        end_idx = start_idx + window_length  # window includes [start_idx, end_idx-1]
        window = x[start_idx:end_idx].astype(np.float32, copy=False)

        inp = np.asarray(window, dtype=np.float32)[None, :]     # (1, context)
        point_forecast, _quantile_forecast = model.forecast(horizon=prediction_length, inputs=inp)

        pf = np.asarray(point_forecast, dtype=np.float32)       # (1, H)
        if not np.isfinite(pf).all():
            raise RuntimeError(
                f"TimesFM produced non-finite forecast. "
                f"window stats: min={float(np.min(inp))}, mean={float(np.mean(inp))}, max={float(np.max(inp))}, "
                f"std={float(np.std(inp))}"
            )

        yhat_h = pf[0]

        yhat_all.append(yhat_h)
        w_start.append(start_idx)
        w_end.append(end_idx - 1)
        x_last.append(float(window[-1]))

    yhat = np.stack(yhat_all, axis=0)  # (N_windows, H)

    return {
        "yhat": yhat,
        "window_start": np.asarray(w_start, dtype=np.int32),
        "window_end": np.asarray(w_end, dtype=np.int32),
        "x_last": np.asarray(x_last, dtype=np.float32),
    }


