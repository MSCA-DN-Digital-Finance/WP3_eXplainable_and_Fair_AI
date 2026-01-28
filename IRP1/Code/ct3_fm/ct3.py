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
    cfg = meta.get("config", {})

    traj = np.load(run_dir / "trajectory.npz")

    pred_dir = run_dir / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(f"Missing predictions directory: {pred_dir}")

    npz_files = list(pred_dir.glob("*.npz"))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No prediction .npz found in {pred_dir}")
    if len(npz_files) > 1:
        raise RuntimeError(
            f"Multiple prediction .npz files found in {pred_dir}: {[p.name for p in npz_files]}"
        )

    preds = np.load(npz_files[0])

    run: Run = {
        "run_dir": run_dir,
        "run_id": run_dir.name,
        "generator": cfg.get("generator", meta.get("generator")),
        "T": cfg.get("T", meta.get("T")),
        "config": cfg,
        "signal": traj["signal"],
        "noise": traj["noise"],
        "x": traj["x"],
        "preds": preds,  # <-- always present, schema depends on model/task
    }

    # Optional convenience fields (only if present)
    if "yhat" in preds:
        run["yhat"] = np.asarray(preds["yhat"], dtype=float)
    if "t_idx" in preds:
        run["t_idx"] = np.asarray(preds["t_idx"], dtype=int)
    if "x_t" in preds:
        run["x_t"] = np.asarray(preds["x_t"], dtype=float)
    if "yhat_q" in preds:
        run["yhat_q"] = np.asarray(preds["yhat_q"], dtype=float)
    if "quantile_levels" in preds:
        run["quantile_levels"] = np.asarray(preds["quantile_levels"], dtype=float)

    return run



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


def metric_beta_hat_model(run: Run, eps: float = 1e-12) -> float:
    """
    Model-implied AR(1) beta:
        beta_hat = sum(x_t * yhat_{t+1}) / sum(x_t^2)

    Measures how strongly the model's forecast depends on the current state.
    Requires:
      - run["x"]     : full level trajectory
      - run["yhat"]  : 1-step-ahead level forecasts
      - run["t_idx"] : indices such that yhat[i] predicts x[t_idx[i] + 1]
    """
    x = np.asarray(run["x"], dtype=float)
    yhat = np.asarray(run["yhat"], dtype=float)
    t_idx = np.asarray(run["t_idx"], dtype=int)

    # x_t aligned with yhat
    x_t = x[t_idx]

    denom = float(np.dot(x_t, x_t))
    if denom < eps:
        return float("nan")

    return float(np.dot(x_t, yhat) / denom)

import numpy as np

def power_spectrum(x, dt=1.0, detrend=True, window=True, normalize=True, nfft=None):
    x = np.asarray(x, float)

    if detrend:
        x = x - np.mean(x)

    if window:
        w = np.hanning(len(x))
        xw = x * w
    else:
        xw = x

    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(len(xw))))  # next pow2 for stable bins

    X = np.fft.rfft(xw, n=nfft)
    P = (np.abs(X) ** 2)
    f = np.fft.rfftfreq(nfft, d=dt)

    if normalize:
        s = P.sum()
        if s > 0:
            P = P / s

    return f, P


def wasserstein_1d_from_spectra(f, P, Q):
    f = np.asarray(f, float)
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)

    P = np.clip(P, 0, None)
    Q = np.clip(Q, 0, None)
    P = P / (P.sum() + 1e-12)
    Q = Q / (Q.sum() + 1e-12)

    cdf_P = np.cumsum(P)
    cdf_Q = np.cumsum(Q)

    df = np.diff(f, prepend=f[0])
    W = np.sum(np.abs(cdf_P - cdf_Q) * df)
    return float(W)

def harmonic_run_spectrum_from_predictions(
    run: Run,
    *,
    quantile: float = 0.5,
    dt: float = 1.0,
    detrend: bool = True,
    window: bool = True,
    normalize: bool = True,
    nfft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (f, P_run) where P_run is the average normalized spectrum over forecast windows.
    """
    preds = run["preds"]  # we'll add this in load_run below
    yhat_q = np.asarray(preds["yhat_q"], dtype=float)          # (W, H, Q)
    q_levels = np.asarray(preds["quantile_levels"], dtype=float)

    # find quantile index (robust)
    q_idx = int(np.argmin(np.abs(q_levels - float(quantile))))
    if abs(q_levels[q_idx] - float(quantile)) > 1e-6:
        raise ValueError(f"Requested quantile={quantile} not found. Available: {q_levels}")

    # compute spectrum for each window forecast path
    P_list = []
    f_ref = None

    for w in range(yhat_q.shape[0]):
        x_fore = yhat_q[w, :, q_idx]  # (H,)
        f, P = power_spectrum(
            x_fore, dt=dt, detrend=detrend, window=window, normalize=normalize, nfft=nfft
        )
        if f_ref is None:
            f_ref = f
        else:
            if len(f) != len(f_ref) or np.max(np.abs(f - f_ref)) > 1e-12:
                raise RuntimeError("Frequency grids differ across windows; fix nfft/dt.")
        P_list.append(P)

    P_run = np.mean(np.stack(P_list, axis=0), axis=0)
    if normalize:
        P_run = P_run / (P_run.sum() + 1e-12)

    return f_ref, P_run



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




def ct3_harmonic_wasserstein_table(
    gen_root: Path,
    *,
    intervention_key: str = "omega",
    quantile: float = 0.5,
    dt: float = 1.0,
    nfft: int | None = None,
) -> pd.DataFrame:
    run_dirs = iter_runs(gen_root)
    runs = [load_run(d) for d in run_dirs]

    # compute one spectrum per theta
    spectra = {}
    for r in runs:
        theta = float(r["config"][intervention_key])
        if theta in spectra:
            raise ValueError(f"Duplicate theta={theta}; aggregate replicates explicitly.")
        f, P = harmonic_run_spectrum_from_predictions(r, quantile=quantile, dt=dt, nfft=nfft)
        spectra[theta] = (f, P)

    thetas = sorted(spectra.keys())
    records = []

    for theta, theta_prime in itertools.product(thetas, thetas):
        if theta == theta_prime:
            continue

        f, P = spectra[theta]
        f2, Q = spectra[theta_prime]

        if len(f) != len(f2) or np.max(np.abs(f - f2)) > 1e-12:
            raise RuntimeError("Frequency grids differ between runs; fix nfft/dt.")

        W = wasserstein_1d_from_spectra(f, P, Q)

        records.append(
            {
                intervention_key: theta,
                f"{intervention_key}_prime": theta_prime,
                "ct3_signed": float("nan"),  # Wasserstein is nonnegative; signed doesn't apply
                "ct3_abs": float(W),
                "divergence": float(W),
            }
        )

    return pd.DataFrame.from_records(records)



# -----------------------
# Optional: convenience runner
# -----------------------

def run_rw_ct3_and_save(
    gen_root: Path,
    *,
    intervention_key: str,
    out_path: Path | None = None,
    metric_name: str = "prob_pos_forecast",
) -> pd.DataFrame:
    
    if out_path is not None and out_path.exists():
        print(f"{out_path}: CT3 output already exists, skipping...")
        return pd.read_csv(out_path)
    
    metric_fn = metric_prob_positive_forecast if metric_name == "prob_pos_forecast" else metric_mean_forecast
    df = ct3_pairwise_table(gen_root, intervention_key=intervention_key, metric_fn=metric_fn)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


def run_ar1_ct3_and_save(
    gen_root: Path,
    *,
    intervention_key: str = "phi",
    out_path: Path | None = None,
    metric_name: str = "beta_hat_model",
) -> pd.DataFrame:
    """
    AR(1) CT3 runner. Uses a model-sensitivity metric by default:
      beta_hat_model = sum(x_t * yhat_{t+1}) / sum(x_t^2)

    metric_name options:
      - "beta_hat_model"
      - "mean_forecast"
    """

    if out_path is not None and out_path.exists():
        print(f"{out_path}: CT3 output already exists, skipping...")
        return pd.read_csv(out_path)
    
    if metric_name == "beta_hat_model":
        metric_fn = metric_beta_hat_model
    elif metric_name == "mean_forecast":
        metric_fn = metric_mean_forecast
    else:
        raise ValueError("metric_name must be 'beta_hat_model' or 'mean_forecast'")

    df = ct3_pairwise_table(gen_root, intervention_key=intervention_key, metric_fn=metric_fn)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df

def run_harmonic_ct3_and_save(
    gen_root: Path,
    *,
    intervention_key: str = "omega",
    out_path: Path | None = None,
    quantile: float = 0.5,
    dt: float = 1.0,
    nfft: int | None = None,
) -> pd.DataFrame:
    if out_path is not None and out_path.exists():
        print(f"{out_path}: CT3 output already exists, skipping...")
        return pd.read_csv(out_path)

    df = ct3_harmonic_wasserstein_table(
        gen_root,
        intervention_key=intervention_key,
        quantile=quantile,
        dt=dt,
        nfft=nfft,
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df


