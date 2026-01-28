from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd


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



