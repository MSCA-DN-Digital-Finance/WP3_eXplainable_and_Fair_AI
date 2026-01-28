# create_preds.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from chronos import Chronos2Pipeline

from models import (
    load_npz_series,
    make_target,
    chronos_rolling_1step_predict,
)


def iter_runs(gen_root: Path):
    for run_dir in sorted(gen_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if (run_dir / "trajectory.npz").exists() and (run_dir / "meta.json").exists():
            yield run_dir


def run_chronos_over_generator(
    pipeline,
    gen_root: Path,
    *,
    series_key: str,
    target_kind: str,
    window_length: int,
):
    for run_dir in iter_runs(gen_root):
        run_id = run_dir.name

        x = load_npz_series(run_dir, key=series_key)
        y = make_target(x, target_kind=target_kind)

        yhat, t_idx = chronos_rolling_1step_predict(
            pipeline,
            y,
            run_id=run_id,
            window_length=window_length,
        )

        pred_dir = run_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)

        np.savez_compressed(
            pred_dir / "chronos2_rolling_1step.npz",
            yhat=yhat,
            t_idx=t_idx,
        )

        meta = {
            "model": "amazon/chronos-2",
            "prediction_length": 1,
            "quantile_levels": [0.5],
            "window_length": window_length,
            "series_key": series_key,
            "target_kind": target_kind,
            "alignment": "yhat[i] predicts y[t_idx[i] + 1]",
        }

        (pred_dir / "chronos2_rolling_1step_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True)
        )

        print(f"{gen_root.name}/{run_id}: saved {len(yhat)} predictions")


if __name__ == "__main__":
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
    )

    run_chronos_over_generator(
        pipeline,
        Path("artifacts/trajectories/rw_drift"),
        series_key="x",
        target_kind="diff",
        window_length=50,
    )