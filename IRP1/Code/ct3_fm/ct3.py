# ct3.py
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import pandas as pd


Run = Dict[str, Any]
MetricFn = Callable[[Run], float]


def iter_runs(gen_root: Path) -> List[Path]:
    return sorted(
        d for d in gen_root.iterdir()
        if d.is_dir() and (d / "trajectory.npz").exists() and (d / "meta.json").exists()
    )


def load_run(run_dir: Path) -> Run:
    meta = json.loads((run_dir / "meta.json").read_text())
    cfg = meta.get("config", {})  # you store intervention values here

    traj = np.load(run_dir / "trajectory.npz")
    pred_path = run_dir / "predictions" / "chronos2_rolling_1step.npz"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions: {pred_path}")

    preds = np.load(pred_path)

    return {
        "run_dir": run_dir,
        "run_id": run_dir.name,
        "generator": cfg.get("generator", meta.get("generator")),
        "T": cfg.get("T", meta.get("T")),
        "config": cfg,
        "signal": traj["signal"],
        "noise": traj["noise"],
        "x": traj["x"],
        "yhat": np.asarray(preds["yhat"], dtype=float),
        "t_idx": np.asarray(preds["t_idx"], dtype=int),
    }


# -----------------------
# Metric functions
# -----------------------

def metric_prob_positive_forecast(run: Run) -> float:
    """
    P(ŷ > 0) estimated as fraction of 1-step forecasts that are positive.
    Appropriate when target_kind="diff" and you care about sign of predicted change.
    """
    yhat = run["yhat"]
    return float(np.mean(yhat > 0.0))


def metric_mean_forecast(run: Run) -> float:
    """Mean of forecasts (sometimes useful as a sanity check)."""
    return float(np.mean(run["yhat"]))


# -----------------------
# CT3 computation
# -----------------------

def compute_scalar_by_theta(
    runs: List[Run],
    intervention_key: str,
    metric_fn: MetricFn,
) -> Dict[float, float]:
    """
    Returns mapping theta_value -> scalar metric m(theta).
    Assumes one run per theta (your minimized sweep). If multiple runs per theta,
    you should average; see note below.
    """
    out: Dict[float, float] = {}
    for r in runs:
        cfg = r["config"]
        if intervention_key not in cfg:
            raise KeyError(f"Run {r['run_id']} missing intervention_key={intervention_key}")
        theta = float(cfg[intervention_key])
        if theta in out:
            raise ValueError(
                f"Duplicate theta={theta} for {intervention_key}. "
                f"If you want replicates, aggregate explicitly."
            )
        out[theta] = metric_fn(r)
    return out


def ct3_pairwise_table(
    gen_root: Path,
    *,
    intervention_key: str,
    metric_fn: MetricFn = metric_prob_positive_forecast,
) -> pd.DataFrame:
    """
    CT3:
      1) compute scalar metric m(theta) for each run
      2) compute pairwise divergence m(theta') - m(theta) over cartesian product
    Returns a DataFrame with columns:
      [theta, theta_prime, ct3_signed, ct3_abs, m_theta, m_theta_prime]
    """
    run_dirs = iter_runs(gen_root)
    runs = [load_run(d) for d in run_dirs]

    m = compute_scalar_by_theta(runs, intervention_key=intervention_key, metric_fn=metric_fn)

    thetas = sorted(m.keys())
    records = []
    for theta, theta_prime in itertools.product(thetas, thetas):
        if theta == theta_prime:
            continue
        m_theta = m[theta]
        m_theta_prime = m[theta_prime]
        delta = m_theta_prime - m_theta
        records.append(
            {
                intervention_key: theta,
                f"{intervention_key}_prime": theta_prime,
                "ct3_signed": float(delta),
                "ct3_abs": float(abs(delta)),
                "m_theta": float(m_theta),
                "m_theta_prime": float(m_theta_prime),
            }
        )

    return pd.DataFrame.from_records(records)


# -----------------------
# Optional: convenience runner
# -----------------------

def run_ct3_and_save(
    gen_root: Path,
    *,
    intervention_key: str,
    out_path: Path | None = None,
    metric_name: str = "prob_pos_forecast",
) -> pd.DataFrame:
    metric_fn = metric_prob_positive_forecast if metric_name == "prob_pos_forecast" else metric_mean_forecast
    df = ct3_pairwise_table(gen_root, intervention_key=intervention_key, metric_fn=metric_fn)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df
