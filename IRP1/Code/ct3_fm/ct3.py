# ct3.py
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: int
    generator: str
    intervention_key: str


EXPERIMENTS = {
    1: ExperimentSpec(1, "rw_drift", "mu"),
    2: ExperimentSpec(2, "ar1", "phi"),
    3: ExperimentSpec(3, "harmonic", "omega"),
    4: ExperimentSpec(4, "ar1", "sigma"),
    5: ExperimentSpec(5, "harmonic", "A"),  # or "amplitude_multiplier" if you prefer
}



Run = Dict[str, Any]
MetricFn = Callable[[Run], float]

# Seed utilities

def get_seed_noise_from_run(run: Run) -> int:
    cfg = run.get("config", {}) or {}
    if "seed_noise" in cfg:
        return int(cfg["seed_noise"])

    meta_seeds = run.get("seeds", {}) or {}
    if "noise" in meta_seeds:
        return int(meta_seeds["noise"])

    raise KeyError(f"Run {run.get('run_id')} missing seed_noise in config and seeds.noise in meta.")
    

def split_runs_by_seed(runs: List[Run]) -> Dict[int, List[Run]]:
    out: Dict[int, List[Run]] = {}
    for r in runs:
        s = get_seed_noise_from_run(r)
        out.setdefault(s, []).append(r)
    return out

# Experiment utilities

def iter_experiment_runs(gen_root: Path, *, model_name: str, experiment_id: int) -> List[Run]:
    run_dirs = iter_runs(gen_root)
    runs = [load_run(d, model_name=model_name) for d in run_dirs]
    out = [r for r in runs if int(r["config"].get("experiment_id", -1)) == int(experiment_id)]
    if not out:
        raise ValueError(f"No runs found for experiment_id={experiment_id} under {gen_root}")
    return out



def iter_runs(gen_root: Path) -> List[Path]:
    return sorted(
        d for d in gen_root.iterdir()
        if d.is_dir() and (d / "trajectory.npz").exists() and (d / "meta.json").exists()
    )


def load_run(run_dir: Path, *, model_name: str) -> Run:
    """
    Load one run (trajectory + predictions) for a specific model.

    Expected layout:
      run_dir/
        trajectory.npz
        meta.json
        predictions/<model_name>/
          preds.npz
          meta.json

    The prediction archive schema can differ by model/task (e.g., yhat/t_idx vs yhat_q).
    This loader is schema-agnostic: it always returns 'preds' and adds convenience keys
    if present.
    """
    # --- run metadata + trajectory ---
    meta = json.loads((run_dir / "meta.json").read_text())
    cfg = meta.get("config", {})

    traj = np.load(run_dir / "trajectory.npz")

    # --- model-specific predictions ---
    pred_dir = run_dir / "predictions" / model_name
    if not pred_dir.exists():
        raise FileNotFoundError(f"Missing predictions directory: {pred_dir}")

    npz_files = sorted(pred_dir.glob("*.npz"))
    json_files = sorted(pred_dir.glob("*.json"))

    if len(npz_files) != 1 or len(json_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 .npz and 1 .json in {pred_dir}, got "
            f"{len(npz_files)} npz and {len(json_files)} json "
            f"(npz={[p.name for p in npz_files]}, json={[p.name for p in json_files]})"
        )

    pred_file = npz_files[0]
    preds = np.load(pred_file)

    run: Run = {
        "run_dir": run_dir,
        "run_id": run_dir.name,
        "generator": cfg.get("generator", meta.get("generator")),
        "T": cfg.get("T", meta.get("T")),
        "config": cfg,
        "noise": traj["noise"],
        "x": traj["x"],
        "model_name": model_name,
        "preds": preds,  # always present; schema depends on model/task
        "seeds": meta.get("seeds", {}),
    }

    # Optional convenience fields (only if present in preds.npz)
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
    if "window_start" in preds:
        run["window_start"] = np.asarray(preds["window_start"], dtype=int)
    if "window_end" in preds:
        run["window_end"] = np.asarray(preds["window_end"], dtype=int)

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
    Compute one normalized power spectrum from a run's harmonic forecasts.

    Supports two prediction formats:
      - Quantile forecasts: yhat_q (W, H, Q) + quantile_levels (Q,)
      - Point forecasts:    yhat   (W, H)

    Strategy:
      - Take the mean forecast trajectory across windows (averaging over W),
        producing a single length-H series, then compute its power spectrum.
    """
    preds = run["preds"]

    # --- extract a single forecast series of length H ---
    if "yhat_q" in preds.files:
        yhat_q = np.asarray(preds["yhat_q"], dtype=float)  # (W, H, Q)

        if "quantile_levels" not in preds.files:
            raise KeyError("Pred archive has yhat_q but missing quantile_levels.")

        q_levels = np.asarray(preds["quantile_levels"], dtype=float)  # (Q,)
        q_idx = int(np.argmin(np.abs(q_levels - float(quantile))))

        # take chosen quantile -> (W, H)
        yhat_wh = yhat_q[:, :, q_idx]

    elif "yhat" in preds.files:
        yhat_wh = np.asarray(preds["yhat"], dtype=float)  # (W, H)

    else:
        raise KeyError(
            f"Pred archive missing required keys. Have: {list(preds.files)}; "
            "expected yhat_q or yhat."
        )

    if yhat_wh.ndim != 2:
        raise ValueError(f"Expected (W, H) after extraction, got shape {yhat_wh.shape}")

    # average across windows -> (H,)
    yhat_h = np.mean(yhat_wh, axis=0)

    # --- compute spectrum ---
    f, P = power_spectrum(
        yhat_h,
        dt=dt,
        detrend=detrend,
        window=window,
        normalize=normalize,
        nfft=nfft,
    )
    return f, P




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
    model_name: str,
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
    runs = [load_run(d, model_name=model_name) for d in run_dirs]

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

def ct3_pairwise_table_across_seeds(
    gen_root: Path,
    *,
    intervention_key: str,
    model_name: str,
    metric_fn: MetricFn,
) -> pd.DataFrame:
    run_dirs = iter_runs(gen_root)
    runs = [load_run(d, model_name=model_name) for d in run_dirs]

    by_seed = split_runs_by_seed(runs)

    per_seed_rows = []
    for seed, seed_runs in sorted(by_seed.items()):
        m = compute_scalar_by_theta(seed_runs, intervention_key=intervention_key, metric_fn=metric_fn)

        thetas = sorted(m.keys())
        for theta, theta_prime in itertools.product(thetas, thetas):
            if theta == theta_prime:
                continue
            delta = m[theta_prime] - m[theta]
            per_seed_rows.append(
                {
                    intervention_key: float(theta),
                    f"{intervention_key}_prime": float(theta_prime),
                    "seed_noise": int(seed),
                    "ct3_signed": float(delta),
                    "ct3_abs": float(abs(delta)),
                    "m_theta": float(m[theta]),
                    "m_theta_prime": float(m[theta_prime]),
                }
            )

    df = pd.DataFrame.from_records(per_seed_rows)

    grp_cols = [intervention_key, f"{intervention_key}_prime"]
    agg = (
        df.groupby(grp_cols, as_index=False)
          .agg(
              ct3_signed_mean=("ct3_signed", "mean"),
              ct3_signed_std=("ct3_signed", "std"),
              ct3_abs_mean=("ct3_abs", "mean"),
              ct3_abs_std=("ct3_abs", "std"),
              n_seeds=("seed_noise", "nunique"),
          )
    )
    return agg


def ct3_pairwise_table_by_experiment_across_seeds(
    gen_root: Path,
    *,
    experiment_id: int,
    model_name: str,
    metric_fn: MetricFn,
    intervention_key: str | None = None,  # optional override
) -> pd.DataFrame:
    runs = iter_experiment_runs(gen_root, model_name=model_name, experiment_id=experiment_id)

    # Determine which parameter is "theta"
    if intervention_key is None:
        intervention_key = EXPERIMENTS[experiment_id].intervention_key

    by_seed = split_runs_by_seed(runs)

    per_seed_rows = []
    for seed, seed_runs in sorted(by_seed.items()):
        # strict uniqueness: one theta per seed per experiment
        m = compute_scalar_by_theta(seed_runs, intervention_key=intervention_key, metric_fn=metric_fn)
        thetas = sorted(m.keys())

        for theta, theta_prime in itertools.product(thetas, thetas):
            if theta == theta_prime:
                continue
            delta = m[theta_prime] - m[theta]
            per_seed_rows.append(
                {
                    "experiment_id": int(experiment_id),
                    intervention_key: float(theta),
                    f"{intervention_key}_prime": float(theta_prime),
                    "seed_noise": int(seed),
                    "ct3_signed": float(delta),
                    "ct3_abs": float(abs(delta)),
                }
            )

    df = pd.DataFrame.from_records(per_seed_rows)

    grp_cols = ["experiment_id", intervention_key, f"{intervention_key}_prime"]
    agg = (
        df.groupby(grp_cols, as_index=False)
          .agg(
              ct3_signed_mean=("ct3_signed", "mean"),
              ct3_signed_std=("ct3_signed", "std"),
              ct3_abs_mean=("ct3_abs", "mean"),
              ct3_abs_std=("ct3_abs", "std"),
              n_seeds=("seed_noise", "nunique"),
          )
    )
    return agg





def ct3_harmonic_wasserstein_table(
    gen_root: Path,
    *,
    intervention_key: str = "omega",
    model_name: str,
    quantile: float = 0.5,
    dt: float = 1.0,
    nfft: int | None = None,
) -> pd.DataFrame:
    run_dirs = iter_runs(gen_root)
    runs = [load_run(d, model_name=model_name) for d in run_dirs]

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


def ct3_harmonic_wasserstein_table_across_seeds(
    gen_root: Path,
    *,
    intervention_key: str = "omega",
    model_name: str,
    quantile: float = 0.5,
    dt: float = 1.0,
    nfft: int | None = None,
) -> pd.DataFrame:
    run_dirs = iter_runs(gen_root)
    runs = [load_run(d, model_name=model_name) for d in run_dirs]

    by_seed = split_runs_by_seed(runs)

    rows = []
    for seed, seed_runs in sorted(by_seed.items()):
        spectra: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

        for r in seed_runs:
            theta = float(r["config"][intervention_key])
            if theta in spectra:
                raise ValueError(f"Duplicate theta={theta} within seed={seed}; expected one run per theta.")
            f, P = harmonic_run_spectrum_from_predictions(r, quantile=quantile, dt=dt, nfft=nfft)
            spectra[theta] = (f, P)

        thetas = sorted(spectra.keys())
        for theta, theta_prime in itertools.product(thetas, thetas):
            if theta == theta_prime:
                continue
            f, P = spectra[theta]
            f2, Q = spectra[theta_prime]

            if len(f) != len(f2) or np.max(np.abs(f - f2)) > 1e-12:
                raise RuntimeError("Frequency grids differ between runs; fix nfft/dt.")

            W = wasserstein_1d_from_spectra(f, P, Q)
            rows.append(
                {
                    intervention_key: float(theta),
                    f"{intervention_key}_prime": float(theta_prime),
                    "seed_noise": int(seed),
                    "ct3_abs": float(W),
                }
            )

    df = pd.DataFrame.from_records(rows)
    grp_cols = [intervention_key, f"{intervention_key}_prime"]
    agg = (
        df.groupby(grp_cols, as_index=False)
          .agg(
              ct3_abs_mean=("ct3_abs", "mean"),
              ct3_abs_std=("ct3_abs", "std"),
              n_seeds=("seed_noise", "nunique"),
          )
    )
    return agg


def ct3_harmonic_wasserstein_by_experiment_across_seeds(
    gen_root: Path,
    *,
    experiment_id: int,
    model_name: str,
    intervention_key: str | None = None,
    quantile: float = 0.5,
    dt: float = 1.0,
    nfft: int | None = None,
) -> pd.DataFrame:
    runs = iter_experiment_runs(gen_root, model_name=model_name, experiment_id=experiment_id)

    if intervention_key is None:
        intervention_key = EXPERIMENTS[experiment_id].intervention_key

    by_seed = split_runs_by_seed(runs)

    rows = []
    for seed, seed_runs in sorted(by_seed.items()):
        spectra: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

        for r in seed_runs:
            theta = float(r["config"][intervention_key])
            if theta in spectra:
                raise ValueError(f"Duplicate theta={theta} within seed={seed} exp={experiment_id}")
            f, P = harmonic_run_spectrum_from_predictions(r, quantile=quantile, dt=dt, nfft=nfft)
            spectra[theta] = (f, P)

        thetas = sorted(spectra.keys())
        for theta, theta_prime in itertools.product(thetas, thetas):
            if theta == theta_prime:
                continue
            f, P = spectra[theta]
            f2, Q = spectra[theta_prime]
            if len(f) != len(f2) or np.max(np.abs(f - f2)) > 1e-12:
                raise RuntimeError("Frequency grids differ; fix nfft/dt.")

            W = wasserstein_1d_from_spectra(f, P, Q)
            rows.append(
                {
                    "experiment_id": int(experiment_id),
                    intervention_key: float(theta),
                    f"{intervention_key}_prime": float(theta_prime),
                    "seed_noise": int(seed),
                    "ct3_abs": float(W),
                }
            )

    df = pd.DataFrame.from_records(rows)
    grp_cols = ["experiment_id", intervention_key, f"{intervention_key}_prime"]
    agg = (
        df.groupby(grp_cols, as_index=False)
          .agg(
              ct3_abs_mean=("ct3_abs", "mean"),
              ct3_abs_std=("ct3_abs", "std"),
              n_seeds=("seed_noise", "nunique"),
          )
    )
    return agg


def ct3_harmonic_wasserstein_by_experiment_across_seeds(
    gen_root: Path,
    *,
    experiment_id: int,
    model_name: str,
    intervention_key: str | None = None,
    quantile: float = 0.5,
    dt: float = 1.0,
    nfft: int | None = None,
) -> pd.DataFrame:
    runs = iter_experiment_runs(gen_root, model_name=model_name, experiment_id=experiment_id)

    if intervention_key is None:
        intervention_key = EXPERIMENTS[experiment_id].intervention_key

    by_seed = split_runs_by_seed(runs)

    rows = []
    for seed, seed_runs in sorted(by_seed.items()):
        spectra: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

        for r in seed_runs:
            theta = float(r["config"][intervention_key])
            if theta in spectra:
                raise ValueError(f"Duplicate theta={theta} within seed={seed} exp={experiment_id}")
            f, P = harmonic_run_spectrum_from_predictions(r, quantile=quantile, dt=dt, nfft=nfft)
            spectra[theta] = (f, P)

        thetas = sorted(spectra.keys())
        for theta, theta_prime in itertools.product(thetas, thetas):
            if theta == theta_prime:
                continue
            f, P = spectra[theta]
            f2, Q = spectra[theta_prime]
            if len(f) != len(f2) or np.max(np.abs(f - f2)) > 1e-12:
                raise RuntimeError("Frequency grids differ; fix nfft/dt.")

            W = wasserstein_1d_from_spectra(f, P, Q)
            rows.append(
                {
                    "experiment_id": int(experiment_id),
                    intervention_key: float(theta),
                    f"{intervention_key}_prime": float(theta_prime),
                    "seed_noise": int(seed),
                    "ct3_abs": float(W),
                }
            )

    df = pd.DataFrame.from_records(rows)
    grp_cols = ["experiment_id", intervention_key, f"{intervention_key}_prime"]
    agg = (
        df.groupby(grp_cols, as_index=False)
          .agg(
              ct3_abs_mean=("ct3_abs", "mean"),
              ct3_abs_std=("ct3_abs", "std"),
              n_seeds=("seed_noise", "nunique"),
          )
    )
    return agg






# -----------------------
# Runner
# -----------------------

def run_rw_ct3_and_save(
    gen_root: Path,
    *,
    exp_id: int,
    intervention_key: str,
    model_name: str,
    out_path: Path | None = None,
    metric_name: str = "prob_pos_forecast",
) -> pd.DataFrame:
    
    if out_path is not None and out_path.exists():
        print(f"{out_path}: CT3 output already exists, skipping...")
        return pd.read_csv(out_path)
    
    metric_fn = metric_prob_positive_forecast if metric_name == "prob_pos_forecast" else metric_mean_forecast
    
    df = ct3_pairwise_table_by_experiment_across_seeds(
        gen_root, 
        experiment_id=exp_id,
        intervention_key=intervention_key, 
        model_name=model_name, 
        metric_fn=metric_fn
        )
    

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


def run_ar1_ct3_and_save(
    gen_root: Path,
    *,
    exp_id: int,
    intervention_key: str = "phi",
    model_name: str,
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

    df = ct3_pairwise_table_by_experiment_across_seeds(
        gen_root, 
        experiment_id=exp_id,
        intervention_key=intervention_key,
        model_name=model_name,
        metric_fn=metric_fn
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df

def run_harmonic_ct3_and_save(
    gen_root: Path,
    *,
    exp_id: int,
    intervention_key: str = "omega",
    model_name: str,
    out_path: Path | None = None,
    quantile: float = 0.5,
    dt: float = 1.0,
    nfft: int | None = None,
) -> pd.DataFrame:
    if out_path is not None and out_path.exists():
        print(f"{out_path}: CT3 output already exists, skipping...")
        return pd.read_csv(out_path)

    df = ct3_harmonic_wasserstein_by_experiment_across_seeds(
        gen_root,
        experiment_id=exp_id,
        intervention_key=intervention_key,
        model_name=model_name,
        quantile=quantile,
        dt=dt,
        nfft=nfft,
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df


