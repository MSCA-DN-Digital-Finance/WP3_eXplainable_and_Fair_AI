from typing import Any, Callable, Dict, List, Sequence, Union, Type, Optional
import numpy as np, copy
from metrics import MetricFn
from custom_generators import GenFactory
from custom_generators import NoiseTrajectoryWrapper


# this class is no longer used; kept for reference: will fail if used as-is
def make_task_factory(
    *,
    task_cls: Type[Any],
    gen_factory: GenFactory,
    agent: Any,
    noise_wrapper_cls: Optional[type] = None,
    **task_params,                       # forwarded to task_cls(...)
):
    """
    Returns: build_all(param_grid, reference_param, task_horizon, n_noise_paths=8, seed=123)
             -> dict with:
                {
                  "eps_paths": [np.ndarray ...],         # K noise vectors (length T)
                  "reference": [Task ...],               # len K
                  "by_mu": { mu: [Task ...] }            # each list len K
                }
    """

    def _wrap_with_noise(gen, eps):
        return noise_wrapper_cls(gen, eps) if noise_wrapper_cls else gen

    def _clone_frozen(a):
        a2 = copy.deepcopy(a)
        if hasattr(a2, "freeze"): a2.freeze()
        if hasattr(a2, "soft_reset"): a2.soft_reset()
        return a2

    def _build_one_task(mu: float, eps: np.ndarray):
        # build generator(s)
        if callable(gen_factory):
            base = gen_factory(mu)
            gen_or_gens = _wrap_with_noise(base, eps)
            kwargs = dict(task_params)
            kwargs.setdefault("generator", gen_or_gens)
        else:
            bases = [gf(mu) for gf in gen_factory]
            gens  = [_wrap_with_noise(g, eps) for g in bases]
            kwargs = dict(task_params)
            kwargs.setdefault("generators", gens)

        # clone/freeze agent per instance
        kwargs.setdefault("agent", _clone_frozen(agent))
        return task_cls(**kwargs)

    def build_all(
        param_grid: Sequence[float],
        reference_param: float,
        task_horizon: int,
        n_noise_paths: int = 8,
        seed: int = 123,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        eps_paths = [rng.normal(0.0, 1.0, size=int(task_horizon)) for _ in range(int(n_noise_paths))]

        # reference tasks (one per eps path)
        reference_tasks = [_build_one_task(reference_param, eps) for eps in eps_paths]

        # tasks per μ, sharing the *same* eps_paths (twin trajectories)
        by_mu: Dict[float, List[Any]] = {}
        for mu in param_grid:
            by_mu[mu] = [_build_one_task(float(mu), eps) for eps in eps_paths]

        return {
            "eps_paths": eps_paths,
            "reference": reference_tasks,
            "by_mu": by_mu,
        }

    return build_all




def make_alloc_task_factory(
    *,
    task_cls,
    gen_factory,         # list of factories [f_const, f_trend, ...]
    agent,
    noise_wrapper_cls=None,
    total_movements: int,
    start_values,
    tc: float,
):
    """
    For AllocationTask, which expects:
        AllocationTask(generators=[g1, g2, ...], agent=..., total_movements=..., ...)
    """

    def build_all(*, param_grid, reference_param, task_horizon, n_noise_paths, seed):
        import random
        rng = np.random.default_rng(seed)
        eps_paths = [rng.normal(0.0, 1.0, size=int(task_horizon)) for _ in range(int(n_noise_paths))]

        def _one_gen_instance(factory, mu, eps_path):
            g = factory(mu)
            if noise_wrapper_cls is None:
                return g
            # use actual NoiseTrajectoryWrapper interface here:
            return noise_wrapper_cls(generator=g, noise_traj=eps_path)  # or whatever it expects

        def _build_one_task(mu, eps_path):
            gens = [_one_gen_instance(f, mu, eps_path) for f in gen_factory]
            return task_cls(
                generators=gens,
                agent=agent,
                total_movements=task_horizon,
                start_values=start_values,
                tc=tc,
            )

        reference_tasks = [_build_one_task(reference_param, eps) for eps in eps_paths]
        by_mu = {mu: [_build_one_task(mu, eps) for eps in eps_paths] for mu in param_grid}

        return {"reference": reference_tasks, "by_mu": by_mu}

    return build_all



def make_pred_task_factory(
    *,
    task_cls,
    gen_factory,         # a single factory: f(mu) -> generator
    agent,
    noise_wrapper_cls=NoiseTrajectoryWrapper,
    total_movements: int,
):
    """
    For PredictionTask, which expects:
        PredictionTask(generator=g, agent=..., total_movements=...)
    """

    def build_all(*, param_grid, reference_param, task_horizon, n_noise_paths, seed):
        import random
        rng = np.random.default_rng(seed)
        eps_paths = [rng.normal(0.0, 1.0, size=int(task_horizon)) for _ in range(int(n_noise_paths))]

        gf = gen_factory  # single factory, not a list

        def _one_gen_instance(mu, eps_path):
            g = gf(mu)
            if noise_wrapper_cls is None:
                return g
            # again, use the actual wrapper signature
            
            return noise_wrapper_cls(generator=g, noise_traj=eps_path)

        def _build_one_task(mu, eps_path):
            gen = _one_gen_instance(mu, eps_path)
            return task_cls(
                generator=gen,
                agent=agent,
                total_movements=task_horizon,
            )

        reference_tasks = [_build_one_task(reference_param, eps_path) for eps_path in eps_paths]
        by_mu = {mu: [_build_one_task(mu, eps_path) for eps_path in eps_paths] for mu in param_grid}

        return {"reference": reference_tasks, "by_mu": by_mu}

    return build_all

