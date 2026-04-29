# create_data.py
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from generation.generators import (
    random_walk_with_drift,
    ar1_with_noise,
    harmonic_oscillator_with_noise,
)

GENERATOR_REGISTRY = {
    "rw_drift": random_walk_with_drift,
    "ar1": ar1_with_noise,
    "harmonic": harmonic_oscillator_with_noise,
}


def stable_hash(obj: Any) -> str:
    """
    Stable hash of JSON-serializable objects (dict/list/str/int/float/bool/None).
    """
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_run(run_dir: Path, traj: Dict[str, Any], config: Dict[str, Any]) -> None:
    ensure_dir(run_dir)

    # Save arrays in compressed npz
    arrays = {
        "noise": np.asarray(traj["noise"], dtype=float),
    }
    if "x" in traj:
        arrays["x"] = np.asarray(traj["x"], dtype=float)

    np.savez_compressed(run_dir / "trajectory.npz", **arrays)

    # Save metadata/config separately (JSON)
    meta = {
        "run_id": run_dir.name,
        "generator": traj.get("name", config["generator"]),
        "T": int(traj.get("T", config["T"])),
        "params": traj.get("params", {}),
        "seeds": traj.get("seeds", {}),
        "config": config,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def grid(params: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    """
    Cartesian product over a dict of lists.
    """
    keys = list(params.keys())
    if not keys:
        yield {}
        return

    def rec(i: int, current: Dict[str, Any]):
        if i == len(keys):
            yield dict(current)
            return
        k = keys[i]
        for v in params[k]:
            current[k] = v
            yield from rec(i + 1, current)

    yield from rec(0, {})




def build_sweep(n_seeds: int = 3, seed_offset: int = 0) -> List[Dict[str, Any]]:
    """
    CT3 assay sweep with replication across noise trajectories.

    - For each experiment condition, repeat with n_seeds independent noise paths.
    - Within each seed, counterfactuals share the same noise realization.
    """

    sweep: List[Dict[str, Any]] = []

    # -----------------------
    # Global settings
    # -----------------------
    T = 1024

    noise_seeds = [seed_offset + i for i in range(n_seeds)]

    TASK_RW = {"task": "single_diff_forecast", "window_length": 200, "forecast_window": 1, "step": 1}
    TASK_AR = {"task": "single_point_forecast", "window_length": 200, "forecast_window": 1, "step": 1}
    TASK_HO = {"task": "multistep_point_forecast", "window_length": 200, "forecast_window": 200, "step": 1}

    # -----------------------
    # Experiment 1: RW drift
    # -----------------------
    sigma_rw = 1.0
    for seed in noise_seeds:
        for mu in [-0.04, -0.02, 0.02, 0.04]:
            sweep.append(
                {
                    "experiment_id": 1,
                    "seed_noise": int(seed),
                    "generator": "rw_drift",
                    "T": T,
                    "x0": 0.0,
                    "mu": float(mu),
                    "sigma": sigma_rw,
                    **TASK_RW,
                }
            )

    # -----------------------
    # Experiment 2: AR(1) persistence
    # -----------------------
    sigma_ar = 1.0
    for seed in noise_seeds:
        for phi in [0.2, 0.5, 0.8, 0.95]:
            sweep.append(
                {
                    "experiment_id": 2,
                    "seed_noise": int(seed),
                    "generator": "ar1",
                    "T": T,
                    "x0": 0.0,
                    "phi": float(phi),
                    "c": 0.0,
                    "sigma": sigma_ar,
                    **TASK_AR,
                }
            )

    # -----------------------
    # Experiment 3: HO wavelength
    # -----------------------
    sigma_ho = 0.05
    fixed_ho = dict(
        dt=1.0,
        phase=0.0,
        gamma=0.0,
        sigma=sigma_ho,
        start_ts=0,
        offset=0.0,
    )

    for seed in noise_seeds:
        for period in [20, 50, 100, 200]:
            omega = float(2 * np.pi / float(period))
            sweep.append(
                {
                    "experiment_id": 3,
                    "seed_noise": int(seed),
                    "generator": "harmonic",
                    "T": T,
                    "A": 1.0,
                    "wavelength": int(period),
                    "omega": omega,
                    **fixed_ho,
                    **TASK_HO,
                }
            )

    # -----------------------
    # Experiment 4: AR(1) noise
    # -----------------------
    phi_fixed = 0.5
    for seed in noise_seeds:
        for sigma in [0.1, 0.3, 1.0, 3.0]:
            sweep.append(
                {
                    "experiment_id": 4,
                    "seed_noise": int(seed),
                    "generator": "ar1",
                    "T": T,
                    "x0": 0.0,
                    "phi": phi_fixed,
                    "c": 0.0,
                    "sigma": float(sigma),
                    **TASK_AR,
                }
            )

    # -----------------------
    # Experiment 5: HO amplitude
    # -----------------------
    period_fixed = 20
    omega_fixed = float(2 * np.pi / float(period_fixed))

    for seed in noise_seeds:
        for mult in [0.25, 0.5, 1.0, 2.0]:
            sweep.append(
                {
                    "experiment_id": 5,
                    "seed_noise": int(seed),
                    "generator": "harmonic",
                    "T": T,
                    "A": float(mult * sigma_ho),
                    "amplitude_multiplier": float(mult),
                    "wavelength": period_fixed,
                    "omega": omega_fixed,
                    **fixed_ho,
                    **TASK_HO,
                }
            )

    return sweep


import inspect
from typing import Any, Dict


def filter_kwargs_for_fn(fn, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs that `fn` accepts (excluding T, which you pass separately)."""
    sig = inspect.signature(fn)
    params = sig.parameters

    # If the function accepts **kwargs, you can pass everything (but here you *don't* want to).
    # We'll still filter to explicit params to avoid leaking metadata.
    allowed = {
        name for name, p in params.items()
        if name != "T" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in cfg.items() if k in allowed}



def run_sweep(out_root: Path) -> None:
    ensure_dir(out_root)

    sweep = build_sweep(n_seeds=10)
    print(f"Planned runs: {len(sweep)}")

    for cfg in sweep:
        gen_name = cfg["generator"]
        T = int(cfg["T"])
        gen_fn = GENERATOR_REGISTRY[gen_name]

        # Only pass args the generator actually accepts
        call_kwargs = filter_kwargs_for_fn(gen_fn, cfg)

        traj = gen_fn(T=T, **call_kwargs)

        run_id = stable_hash(cfg)  # still hashes full config (incl. seed/task/experiment_id)
        run_dir = out_root / gen_name / run_id

        if (run_dir / "trajectory.npz").exists() and (run_dir / "meta.json").exists():
            continue

        save_run(run_dir=run_dir, traj=traj, config=cfg)

    print("Done.")



if __name__ == "__main__":
    out_root = Path("artifacts/trajectories")
    run_sweep(out_root)
