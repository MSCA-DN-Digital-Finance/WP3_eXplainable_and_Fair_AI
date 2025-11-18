# ct3_single.py
from __future__ import annotations
import json, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

TaskLog  = List[Dict[str, Any]]
MetricFn = Callable[[TaskLog, TaskLog], Dict[str, float]]

@dataclass
class CT3Config:
    # identifiers for folder structure
    task_name: str
    agent_name: str
    generator_name: str
    # how to build tasks (your make_task_factory(...) output)
    build_all: Callable[..., Dict[str, Any]]
    # CT3 knobs
    param_grid: Sequence[float]
    reference_param: float
    task_horizon: int
    n_noise_paths: int = 8
    seed: int = 123
    # metrics (return dicts, e.g. {"alloc_L1":..., "alloc_response":...})
    metrics: Sequence[MetricFn] = ()
    # io
    out_root: str | Path = "results"

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

class CT3SingleExecutor:
    """
    Minimal CT3 executor:
      - builds tasks via cfg.build_all(...)
      - runs each (k, μ) rollout
      - computes metrics(ref_log, mu_log) per μ
      - writes one file per k:
          {out_root}/{timestamp}/{task}/{agent}/{generator}/k={k}/metrics.json
    Assumes build_all(...) returns:
        {
          "reference": [Task_k for k in 0..K-1],
          μ1: [Task_k ...],
          μ2: [...], ...
        }
    """
    def __init__(self, cfg: CT3Config, timestamp: str | None = None):
        self.cfg = cfg
        self.timestamp = timestamp or time.strftime("%Y%m%d-%H%M%S")

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        # 1) Build all tasks for this CT3 config
        bundle = cfg.build_all(
            param_grid=list(cfg.param_grid),
            reference_param=cfg.reference_param,
            task_horizon=cfg.task_horizon,
            n_noise_paths=cfg.n_noise_paths,
            seed=cfg.seed,
        )
        # Basic checks
        assert "reference" in bundle, "build_all must return key 'reference'."
        K = len(bundle["reference"])
        assert K == cfg.n_noise_paths, "n_noise_paths mismatch between cfg and factory output."

        # 2) For each noise path k: run ref + each μ, compute metrics, write one file
        root = Path(cfg.out_root) / self.timestamp / cfg.task_name / cfg.agent_name / cfg.generator_name
        _ensure_dir(root)
        artifacts: Dict[str, str] = {}

        # run reference rollouts once per k and cache logs
        ref_logs: List[TaskLog] = []
        for k in range(K):
            t_ref = bundle["reference"][k]
            _ = t_ref.play_game()
            ref_logs.append(t_ref.log)

        by_mu = bundle["by_mu"]

        # now evaluate per μ
        for k in range(K):
            k_dir = root / f"k={k}"
            _ensure_dir(k_dir)
            mu_metrics: Dict[str, Dict[str, float]] = {}

            for mu in cfg.param_grid:
                t_mu = by_mu[mu][k]
                _ = t_mu.play_game()
                ref_log = ref_logs[k]
                mu_log  = t_mu.log

                # write task log to file
                log_path = k_dir / f"task_logs[{mu}].json"
                _write_json(log_path, mu_log)

                # merge metrics from all metric functions
                result: Dict[str, float] = {}
                for mf in cfg.metrics:
                    out = mf(ref_log, mu_log)
                    if not isinstance(out, dict):
                        raise ValueError("MetricFn must return a dict[str, float]")
                    result.update(out)

                mu_metrics[str(mu)] = result

            # minimal metadata + metrics
            payload = {
                "timestamp": self.timestamp,
                "task": cfg.task_name,
                "agent": cfg.agent_name,
                "generator": cfg.generator_name,
                "param_grid": list(map(float, cfg.param_grid)),
                "reference_param": float(cfg.reference_param),
                "task_horizon": int(cfg.task_horizon),
                "n_noise_paths": int(cfg.n_noise_paths),
                "seed": int(cfg.seed),
                "metrics_per_mu": mu_metrics,  # <- core output for this k
            }
            out_path = k_dir / "metrics.json"
            _write_json(out_path, payload)
            artifacts[f"k={k}"] = str(out_path.resolve())

            

        # Also write a tiny config snapshot at the root folder (optional, helpful)
        cfg_path = root / "config_snapshot.json"
        _write_json(cfg_path, {
            "timestamp": self.timestamp,
            "task": cfg.task_name,
            "agent": cfg.agent_name,
            "generator": cfg.generator_name,
            "param_grid": list(map(float, cfg.param_grid)),
            "reference_param": float(cfg.reference_param),
            "task_horizon": int(cfg.task_horizon),
            "n_noise_paths": int(cfg.n_noise_paths),
            "seed": int(cfg.seed),
            "metric_fns": [getattr(m, "__name__", "metric") for m in cfg.metrics],
        })

        return {"root": str(root.resolve()), "artifacts": artifacts, "config": str(cfg_path.resolve())}
