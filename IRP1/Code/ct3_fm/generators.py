# generators.py
# Minimal trajectory generators with *separated* noise trajectories.
# - Each generator returns a "signal" and a "noise" component (both length T),
#   plus "x" which is signal + noise (for convenience).
# - Noise is generated independently so you can later do counterfactual swaps
#   (e.g., same signal, different noise; same noise, different signal).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np


Array = np.ndarray


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def gaussian_noise(T: int, sigma: float, seed: int) -> Array:
    """
    Independent Gaussian noise trajectory eps_t ~ N(0, sigma^2).
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")
    r = _rng(seed)
    return r.normal(loc=0.0, scale=float(sigma), size=T)


def random_walk_with_drift(
    T: int,
    *,
    x0: float = 0.0,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed_signal: int = 0,
    seed_noise: int = 1,
    return_x: bool = True,
) -> Dict[str, Any]:
    """
    Random walk with drift (signal) + independent noise:
        signal: s_{t+1} = s_t + mu
        noise:  eps_t ~ N(0, sigma^2)
        x_t = s_t + eps_t

    Note: This is a drift-only signal; all randomness is in noise so you can
    counterfactually swap noise trajectories later.
    """
    if T <= 0:
        raise ValueError("T must be positive.")

    # deterministic signal given parameters
    t = np.arange(T, dtype=float)
    signal = float(x0) + float(mu) * t

    noise = gaussian_noise(T=T, sigma=sigma, seed=seed_noise)

    out = {
        "name": "rw_drift",
        "T": int(T),
        "params": {"x0": float(x0), "mu": float(mu), "sigma": float(sigma)},
        "seeds": {"signal": int(seed_signal), "noise": int(seed_noise)},
        "signal": signal.astype(float),
        "noise": noise.astype(float),
    }
    if return_x:
        out["x"] = (out["signal"] + out["noise"]).astype(float)
    return out


def ar1_with_noise(
    T: int,
    *,
    x0: float = 0.0,
    phi: float = 0.9,
    c: float = 0.0,
    sigma: float = 1.0,
    seed_noise: int = 1,
    return_x: bool = True,
) -> Dict[str, Any]:
    """
    AR(1) (deterministic recursion) + independent noise:
        noise:  eps_t ~ N(0, sigma^2)
        x_t = c + phi * x_t + eps_t

    Again: all randomness is in the separate noise trajectory.
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if not (-1.0 <= phi <= 1.0):
        # you may allow outside [-1,1], but it's usually unstable
        raise ValueError("phi should be within [-1, 1] for a stable AR(1).")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")

    

    noise = gaussian_noise(T=T, sigma=sigma, seed=seed_noise)

    out = {
        "name": "ar1",
        "T": int(T),
        "params": {"x0": float(x0), "phi": float(phi), "c": float(c), "sigma": float(sigma)},
        "seeds": {"noise": int(seed_noise)},
        "noise": noise,
    }
    if return_x:
        x = np.empty(T, dtype=float)
        x[0] = float(x0)
        for t in range(T - 1):
            x[t + 1] = float(c) + float(phi) * x[t] + noise[t]
        out["x"] = x.astype(float)
    return out


def harmonic_oscillator_with_noise(
    T: int,
    *,
    A: float = 1.0,
    omega: float = 2.0 * np.pi / 50.0,
    phase: float = 0.0,
    offset: float = 0.0,
    sigma: float = 0.1,
    seed_noise: int = 1,
    return_x: bool = True,
) -> Dict[str, Any]:
    """
    Harmonic oscillator (sinusoid) + independent noise:
        signal: s_t = offset + A * sin(omega * t + phase)
        noise:  eps_t ~ N(0, sigma^2)
        x_t = s_t + eps_t
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")
    if A < 0:
        raise ValueError("A should be >= 0 (use phase shift if you want sign flips).")

    t = np.arange(T, dtype=float)
    signal = float(offset) + float(A) * np.sin(float(omega) * t + float(phase))

    noise = gaussian_noise(T=T, sigma=sigma, seed=seed_noise)

    out = {
        "name": "harmonic",
        "T": int(T),
        "params": {
            "A": float(A),
            "omega": float(omega),
            "phase": float(phase),
            "offset": float(offset),
            "sigma": float(sigma),
        },
        "seeds": {"noise": int(seed_noise)},
        "signal": signal.astype(float),
        "noise": noise.astype(float),
    }
    if return_x:
        out["x"] = (out["signal"] + out["noise"]).astype(float)
    return out


# Optional: a simple registry + factory (useful for experiment configs)
GENERATOR_REGISTRY = {
    "rw_drift": random_walk_with_drift,
    "ar1": ar1_with_noise,
    "harmonic": harmonic_oscillator_with_noise,
}


def make_generator(name: str):
    try:
        return GENERATOR_REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"Unknown generator '{name}'. Available: {list(GENERATOR_REGISTRY)}") from e


# Optional: counterfactual utility helpers
def swap_noise(traj: Dict[str, Any], new_noise: Array) -> Dict[str, Any]:
    """
    Return a shallow-copied trajectory dict where 'noise' is replaced and 'x' recomputed.
    """
    if "signal" not in traj:
        raise ValueError("traj must contain 'signal'.")
    if len(new_noise) != len(traj["signal"]):
        raise ValueError("new_noise must have same length as traj['signal'].")

    out = dict(traj)
    out["noise"] = np.asarray(new_noise, dtype=float)
    out["x"] = (np.asarray(out["signal"], dtype=float) + out["noise"]).astype(float)
    return out
