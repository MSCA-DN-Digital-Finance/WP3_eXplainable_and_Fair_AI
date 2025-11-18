# ct3_orchestrator.py
from __future__ import annotations
import time, json
from pathlib import Path
from typing import Sequence, Dict, Any
from ct3_single_run import CT3Config, CT3SingleExecutor

def run_suite(configs: Sequence[CT3Config]) -> Dict[str, Any]:
    """
    Runs a list of CT3 configs under a shared timestamp root.
    Returns a summary with artifact paths.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    summary: Dict[str, Any] = {"timestamp": ts, "runs": {}}
    for cfg in configs:
        ex = CT3SingleExecutor(cfg, timestamp=ts)
        res = ex.run()
        summary["runs"][f"{cfg.task_name}/{cfg.agent_name}/{cfg.generator_name}"] = res

    # write top-level summary next to first config's out_root
    if configs:
        out_root = Path(configs[0].out_root) / ts
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary_path"] = str((out_root / "summary.json").resolve())
    return summary



