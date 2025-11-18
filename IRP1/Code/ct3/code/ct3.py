import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from typing import Sequence, Any, Dict, List
from metrics import MetricFn

class CT3Evaluator:
    """
    Counterfactual Twin–Trajectory Test (CT3) evaluator.

    Parameters
    ----------
    generator_factory : Callable[[float], object]
        Function that creates a *base* generator given a parameter value (e.g., mu).
        Example: lambda mu: LinearTrendGenerator(slope=mu)

    noise_wrapper_cls : type
        A wrapper class that takes (base_generator, eps) and exposes:
          - generate_value(last_value)
          - reset()  -> resets internal noise index to 0
        Example: NoiseTrajectoryWrapper

    agent_frozen : object
        A *trained* agent to be cloned for evaluation.
        Expected optional methods/attrs:
          - freeze()       : disable learning during observe()
          - soft_reset()   : clear buffers without losing weights
          - window_size    : (int) used to set a default warmup
          - action_distribution() -> np.array([P(0), P(1)])  [optional]
          - observe(value), place_bet()

    task_horizon : int
        Number of steps per rollout (T).

    warmup : int or None
        Number of initial steps to exclude from metric calculations (buffer/warmup).
        If None and `agent_frozen` has `window_size`, that is used.

    param_grid : list[float]
        List of parameter values to evaluate (e.g., mus).
        Should include the reference parameter if you want response vs reference.

    reference_param : float
        The reference parameter (e.g., training mu) for delta and divergence.

    n_noise_paths : int
        Number of independent fixed-noise trajectories (K) to average over.

    seed : int or None
        RNG seed for reproducibility.

    start_value : float
        Initial value for the time series.

    Notes
    -----
    - Uses a custom rollout loop (instead of PredictionTask) so we can record
      per-step action distributions if the agent provides them.
    - If `action_distribution()` is unavailable, JS divergence will be skipped and
      only action-based metrics are returned.
    """

    def __init__(
        self,
        generator_factory,
        noise_wrapper_cls,
        agent_frozen,
        task_horizon=1000,
        warmup=None,
        param_grid=None,
        reference_param=0.0,
        n_noise_paths=8,
        seed=123,
        start_value=0.0,
    ):
        self.generator_factory = generator_factory
        self.noise_wrapper_cls = noise_wrapper_cls
        self.agent_template = agent_frozen
        self.T = int(task_horizon)
        self.start_value = float(start_value)

        if warmup is None and hasattr(agent_frozen, "window_size"):
            warmup = int(getattr(agent_frozen, "window_size"))
        self.warmup = int(warmup) if warmup is not None else 0

        if param_grid is None:
            raise ValueError("param_grid must be provided (list of parameter values).")
        self.param_grid = list(param_grid)
        self.reference_param = float(reference_param)

        self.K = int(n_noise_paths)
        self.rng = np.random.default_rng(seed)

        # results containers
        self.results = None   # dict[param] -> metrics dict with lists per noise path
        self.agg = None       # aggregated means/CI per param

    # ---------- Utilities ----------

    @staticmethod
    def _clone_frozen(agent):
        a = copy.deepcopy(agent)
        if hasattr(a, "freeze"):
            a.freeze()
        if hasattr(a, "soft_reset"):
            a.soft_reset()
        return a

    @staticmethod
    def _js_divergence_batch(P, Q, eps=1e-12):
        """
        Jensen–Shannon divergence averaged over rows.
        P, Q: shape (N, C) probability arrays; C=2 here.
        """
        P = np.clip(P, eps, 1.0 - eps)
        Q = np.clip(Q, eps, 1.0 - eps)
        M = 0.5 * (P + Q)
        kl_PM = np.sum(P * (np.log(P) - np.log(M)), axis=1)
        kl_QM = np.sum(Q * (np.log(Q) - np.log(M)), axis=1)
        js = 0.5 * (kl_PM + kl_QM)
        return float(np.mean(js))

    @staticmethod
    def _hamming(a, b):
        return float(np.mean(a != b))

    @staticmethod
    def _mean_ci(x, alpha=0.05):
        x = np.asarray(x, dtype=float)
        m = float(np.mean(x))
        s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        z = 1.96  # ~95% normal approx
        ci = z * (s / math.sqrt(len(x))) if len(x) > 1 else 0.0
        return m, (m - ci, m + ci)

    # ---------- Core rollout ----------

    def _rollout(self, agent, gen):
        """
        One rollout of length T. Returns:
          probs: (T, 2) array of [P(0), P(1)] or None if unavailable
          acts : (T,)   array of hard actions {0,1}
        """
        last = self.start_value
        T = self.T
        probs = []
        acts = []

        # ensure clean state
        if hasattr(gen, "reset"):
            gen.reset()
        if hasattr(agent, "soft_reset"):
            agent.soft_reset()

        # run loop
        for _ in range(T):
            # Observe last value (should not learn if agent is frozen)
            if hasattr(agent, "observe"):
                agent.observe(last)

            # Get probability distribution if available
            p = None
            if hasattr(agent, "action_distribution"):
                try:
                    p = np.asarray(agent.action_distribution(), dtype=float)
                    if p.ndim == 1:
                        p = p.reshape(1, -1)
                except Exception:
                    p = None

            # Hard action
            a = int(agent.place_bet()) if hasattr(agent, "place_bet") else 0

            # Generate next value
            val = gen.generate_value(last)
            last = val

            # Store
            acts.append(a)
            if p is not None and p.shape[-1] == 2:
                probs.append(p[0])
            else:
                probs.append(None)

        # Stack results
        acts = np.asarray(acts, dtype=int)
        if all(x is not None for x in probs):
            probs = np.vstack(probs).astype(float)
        else:
            probs = None

        return probs, acts

    # ---------- Public API ----------

    def run(self):
        """
        Execute CT3 across the parameter grid, using the reference_param as baseline.
        Returns a dict of raw metrics (lists per noise path) and stores aggregated stats.
        """
        # Containers: per param, store lists over K noise paths
        res = {
            mu: {
                "D_ham": [],
                "delta_p": [],
                "JS": [],        # may remain empty if probs unavailable
                "p_up": [],      # absolute p_up under mu (post-warmup)
            }
            for mu in self.param_grid
        }

        for k in range(self.K):
            # sample noise path for this replicate
            eps = self.rng.normal(loc=0.0, scale=1.0, size=self.T)

            # reference rollout
            gen_ref = self.noise_wrapper_cls(self.generator_factory(self.reference_param), eps)
            agent_ref = self._clone_frozen(self.agent_template)
            probs_ref, acts_ref = self._rollout(agent_ref, gen_ref)

            sl = slice(self.warmup, self.T)
            acts_ref_sl = acts_ref[sl]
            probs_ref_sl = probs_ref[sl] if probs_ref is not None else None

            for mu in self.param_grid:
                gen = self.noise_wrapper_cls(self.generator_factory(mu), eps)
                agent = self._clone_frozen(self.agent_template)
                probs, acts = self._rollout(agent, gen)

                acts_sl = acts[sl]
                # Hamming vs reference
                D_ham = self._hamming(acts_sl, acts_ref_sl)
                res[mu]["D_ham"].append(D_ham)

                # Delta p_up
                p_up_mu  = float(np.mean(acts_sl == 1))
                p_up_ref = float(np.mean(acts_ref_sl == 1))
                res[mu]["delta_p"].append(p_up_mu - p_up_ref)
                res[mu]["p_up"].append(p_up_mu)

                # JS divergence on probabilities if available
                if probs is not None and probs_ref_sl is not None:
                    probs_sl = probs[sl]
                    JS = self._js_divergence_batch(probs_sl, probs_ref_sl)
                    res[mu]["JS"].append(JS)

        # Aggregate means & 95% CI
        agg = {}
        for mu, d in res.items():
            mu_stats = {}
            for key, vals in d.items():
                if len(vals) == 0:
                    continue
                mean, (lo, hi) = self._mean_ci(vals)
                mu_stats[key] = {"mean": mean, "ci95": (lo, hi)}
            agg[mu] = mu_stats

        self.results = res
        self.agg = agg
        return res

    def plot_response(self, show=True):
        """
        Plot causal response curves:
          - Δp_up(μ) vs μ  (directional shift)
          - D_ham(μ) vs μ  (binary sensitivity)
          - JS(μ) vs μ     (if probabilities are available)
        """
        if self.agg is None:
            raise RuntimeError("Call run() before plot_response().")

        mus = sorted(self.agg.keys())
        def get_series(key):
            m, lo, hi = [], [], []
            for mu in mus:
                entry = self.agg[mu].get(key, None)
                if entry is None:
                    m.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                else:
                    m.append(entry["mean"])
                    lo.append(entry["ci95"][0])
                    hi.append(entry["ci95"][1])
            return np.array(m), np.array(lo), np.array(hi)

        delta_m, delta_lo, delta_hi = get_series("delta_p")
        dham_m, dham_lo, dham_hi   = get_series("D_ham")
        js_m, js_lo, js_hi         = get_series("JS")

        fig, ax = plt.subplots(figsize=(7, 5))

        # Δp_up(μ)
        ax.errorbar(mus, delta_m, yerr=[delta_m - delta_lo, delta_hi - delta_m],
                    fmt='o-', capsize=3, label='Δ p_up (vs reference)')

        # D_ham(μ)
        ax.errorbar(mus, dham_m, yerr=[dham_m - dham_lo, dham_hi - dham_m],
                    fmt='s--', capsize=3, label='D_ham (vs reference)')

        # JS(μ) if available
        if not np.all(np.isnan(js_m)):
            ax.errorbar(mus, js_m, yerr=[js_m - js_lo, js_hi - js_m],
                        fmt='^-.', capsize=3, label='JS divergence (vs reference)')

        ax.axvline(self.reference_param, linestyle=':', linewidth=1)
        ax.set_xlabel("Generator parameter (μ)")
        ax.set_ylabel("Behavioral change (avg over time, post-warmup)")
        ax.set_title("CT3 Causal Response Curves")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax
    







class CT3Runner:
    def __init__(
        self,
        build_all,                      # from make_task_factory(...)
        param_grid: Sequence[float],
        reference_param: float,
        task_horizon: int,
        n_noise_paths: int = 8,
        seed: int = 123,
        metrics: List[MetricFn] = (),
    ):
        self.build_all = build_all
        self.param_grid = list(param_grid)
        self.reference_param = float(reference_param)
        self.T = int(task_horizon)
        self.K = int(n_noise_paths)
        self.seed = int(seed)
        self.metrics = list(metrics)

        self.results = None   # raw per-metric lists
        self.agg = None       # mean & 95% CI

    @staticmethod
    def _mean_ci(x, z=1.96):
        x = np.asarray(x, float)
        if len(x) == 0:
            return np.nan, (np.nan, np.nan)
        m = float(x.mean())
        s = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        ci = z * (s / math.sqrt(len(x))) if len(x) > 1 else 0.0
        return m, (m - ci, m + ci)

    def run(self):
        bundle = self.build_all(
            param_grid=self.param_grid,
            reference_param=self.reference_param,
            task_horizon=self.T,
            n_noise_paths=self.K,
            seed=self.seed,
        )
        ref_tasks: List[Any] = bundle["reference"]
        by_mu: Dict[float, List[Any]] = bundle["by_mu"]

        # discover metric names on first evaluation
        metric_names = None
        res = {mu: {} for mu in self.param_grid}

        for k in range(self.K):
            # run reference path k
            ref_task = ref_tasks[k]
            ref_task.play_game()
            ref_log = ref_task.log

            # run each mu on same eps path k
            for mu in self.param_grid:
                t = by_mu[mu][k]
                t.play_game()
                mu_log = t.log

                out = {}
                for fn in self.metrics:
                    out.update(fn(ref_log, mu_log))

                if metric_names is None:
                    metric_names = sorted(out.keys())
                    for m in self.param_grid:
                        res[m] = {name: [] for name in metric_names}

                for name in metric_names:
                    res[mu][name].append(float(out[name]))

        # aggregate
        agg = {}
        for mu, d in res.items():
            stats = {}
            for name, vals in d.items():
                m, (lo, hi) = self._mean_ci(vals)
                stats[name] = {"mean": m, "ci95": (lo, hi)}
            agg[mu] = stats

        self.results = res
        self.agg = agg
        return res
