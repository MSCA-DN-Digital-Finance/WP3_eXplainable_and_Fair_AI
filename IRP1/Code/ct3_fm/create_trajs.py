# create_data.py
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from generators import (
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
        "signal": np.asarray(traj["signal"], dtype=float),
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


def build_sweep() -> List[Dict[str, Any]]:
    """
    Minimal sweep: 4 intervention values per generator parameter (mu, phi, omega),
    everything else fixed. Single T.
    """
    sweep: List[Dict[str, Any]] = []

    T = 1024
    seed_noise = 42  # fixed so only the intervention parameter changes
    sigma = 1.0      # fixed noise scale

    # 4 interventions on mu (rw_drift)
    for mu in [-0.02, -0.01, 0.01, 0.02]:
        sweep.append(
            {
                "generator": "rw_drift",
                "T": T,
                "x0": 0.0,
                "mu": mu,
                "sigma": sigma,
                "seed_noise": seed_noise,
            }
        )

    # 4 interventions on phi (ar1)
    for phi in [0.2, 0.5, 0.8, 0.95]:
        sweep.append(
            {
                "generator": "ar1",
                "T": T,
                "x0": 0.0,
                "phi": phi,
                "c": 0.0,
                "sigma": sigma,
                "seed_noise": seed_noise,
            }
        )

    # 4 interventions on omega (harmonic)
    for omega in [2 * np.pi / 20.0, 2 * np.pi / 50.0, 2 * np.pi / 100.0, 2 * np.pi / 200.0]:
        sweep.append(
            {
                "generator": "harmonic",
                "T": T,
                "A": 1.0,
                "omega": float(omega),
                "phase": 0.0,
                "offset": 0.0,
                "sigma": sigma,
                "seed_noise": seed_noise,
            }
        )

    return sweep


def run_sweep(out_root: Path) -> None:
    ensure_dir(out_root)

    sweep = build_sweep()
    print(f"Planned runs: {len(sweep)}")

    for cfg in sweep:
        gen_name = cfg["generator"]
        T = int(cfg["T"])
        gen_fn = GENERATOR_REGISTRY[gen_name]

        # IMPORTANT: exclude non-generator keys before calling
        call_kwargs = {k: v for k, v in cfg.items() if k not in {"generator", "T"}}

        traj = gen_fn(T=T, **call_kwargs)

        # stable run id so the directory name is deterministic
        run_id = stable_hash(cfg)
        run_dir = out_root / gen_name / run_id

        # skip if exists (idempotent)
        if (run_dir / "trajectory.npz").exists() and (run_dir / "meta.json").exists():
            continue

        save_run(run_dir=run_dir, traj=traj, config=cfg)

    print("Done.")


if __name__ == "__main__":
    out_root = Path("artifacts/trajectories")
    run_sweep(out_root)
