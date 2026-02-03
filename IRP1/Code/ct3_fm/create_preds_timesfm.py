# create_preds.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import timesfm

from models import *
from artifacts_io import *

def iter_runs(gen_root: Path):
    for run_dir in sorted(gen_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if (run_dir / "trajectory.npz").exists() and (run_dir / "meta.json").exists():
            yield run_dir




# TimesFM over generator functions

def run_timesfm_over_rw(
    model,
    gen_root: Path,
    *,
    series_key: str,
    target_kind: str,
    window_length: int,
):
    for run_dir in iter_runs(gen_root):
        run_id = run_dir.name

        pred_dir = run_dir / "predictions" / "timesfm"

        # Hardened skip/recompute logic:
        # - skip only if exactly the expected files exist AND yhat is finite
        # - if artifacts exist but are invalid/partial/wrong-name -> delete and recompute
        skip, pred_file, meta_file = should_skip_or_recompute(
            pred_dir,
            expected_npz_name="timesfm_rolling_1step.npz",
            expected_json_name="timesfm_rolling_1step_meta.json",
            required_keys=("yhat", "t_idx"),
            finite_keys=("yhat",),
            shape_constraints={
                "yhat": (None,),
                "t_idx": (None,),
            },
        )
        if skip:
            print(f"{gen_root.name}/{run_id}: predictions valid, skipping...")
            continue

        x = load_npz_series(run_dir, key=series_key)
        y = make_target(x, target_kind=target_kind)

        yhat, t_idx = timesfm_rolling_1step_predict(
            model,
            y,
            window_length=window_length,
        )

        # Hard guard: never persist NaN/Inf artifacts
        if not np.isfinite(yhat).all():
            raise RuntimeError(f"{gen_root.name}/{run_id}: TimesFM produced non-finite yhat; aborting.")

        np.savez_compressed(
            pred_file,
            yhat=yhat,
            t_idx=t_idx,
        )

        meta = {
            "model": "google/timesfm-2.5-200m-pytorch",
            "prediction_length": 1,
            "window_length": window_length,
            "series_key": series_key,
            "target_kind": target_kind,
            "alignment": "yhat[i] predicts y[t_idx[i] + 1]",
        }

        meta_file.write_text(json.dumps(meta, indent=2, sort_keys=True))

        print(f"{gen_root.name}/{run_id}: saved {len(yhat)} predictions")


def run_timesfm_over_ar1(
    model,
    gen_root: Path,
    *,
    window_length: int = 50,
    series_key: str = "x",
):
    for run_dir in sorted(d for d in gen_root.iterdir() if d.is_dir()):
        if not (run_dir / "trajectory.npz").exists():
            continue

        run_id = run_dir.name
        pred_dir = run_dir / "predictions" / "timesfm"

        # Hardened skip/recompute logic:
        skip, pred_file, meta_file = should_skip_or_recompute(
            pred_dir,
            expected_npz_name="timesfm_ar1_levels_rolling_1step.npz",
            expected_json_name="timesfm_ar1_levels_rolling_1step_meta.json",
            required_keys=("yhat", "t_idx", "x_last"),
            finite_keys=("yhat", "x_last"),
            shape_constraints={
                "yhat": (None,),
                "t_idx": (None,),
                "x_last": (None,),
            },
        )
        if skip:
            print(f"{gen_root.name}/{run_id}: predictions valid, skipping...")
            continue

        x = load_npz_series(run_dir, key=series_key)

        # TimesFM forecasts directly on levels for AR(1)
        yhat, t_idx = timesfm_rolling_1step_predict(
            model,
            x,
            window_length=window_length,
        )

        # last observed x_t aligned with t_idx
        x_last = x[t_idx].astype(np.float32, copy=False)

        # Hard guards: never persist NaN/Inf artifacts
        if not np.isfinite(yhat).all():
            raise RuntimeError(f"{gen_root.name}/{run_id}: TimesFM produced non-finite yhat; aborting.")
        if not np.isfinite(x_last).all():
            raise RuntimeError(f"{gen_root.name}/{run_id}: x_last contains non-finite values; aborting.")

        np.savez_compressed(
            pred_file,
            yhat=yhat,
            t_idx=t_idx,
            x_last=x_last,
        )

        meta = {
            "model": "google/timesfm-2.5-200m-pytorch",
            "task": "ar1_levels_rolling_1step",
            "prediction_length": 1,
            "window_length": window_length,
            "series_key": series_key,
            "alignment": "yhat[i] predicts x[t_idx[i] + 1]",
        }
        meta_file.write_text(json.dumps(meta, indent=2, sort_keys=True))

        print(f"ar1/{run_id}: saved {len(yhat)} forecasts")



def run_timesfm_over_harmonic(
    model,
    gen_root: Path,
    *,
    series_key: str = "x",
    window_length: int = 100,
    prediction_length: int = 200,
    step_size: int = 1,
):
    for run_dir in iter_runs(gen_root):
        run_id = run_dir.name
        pred_dir = run_dir / "predictions" / "timesfm"

        # Hardened skip/recompute logic:
        skip, pred_file, meta_file = should_skip_or_recompute(
            pred_dir,
            expected_npz_name="timesfm_harmonic_multistep.npz",
            expected_json_name="timesfm_harmonic_multistep_meta.json",
            required_keys=("yhat", "window_start", "window_end", "x_last"),
            finite_keys=("yhat", "x_last"),
            shape_constraints={
                "yhat": (None, prediction_length),  # (W, H)
                "window_start": (None,),
                "window_end": (None,),
                "x_last": (None,),
            },
        )
        if skip:
            print(f"{gen_root.name}/{run_id}: predictions valid, skipping...")
            continue

        x = load_npz_series(run_dir, key=series_key)

        # TimesFM multistep point forecasts (levels)
        out = timesfm_harmonic_multistep_forecast(
            model,
            x,
            window_length=window_length,
            prediction_length=prediction_length,
            step_size=step_size,
        )
        # expected keys in out:
        #   yhat: (W, H)
        #   window_start: (W,)
        #   window_end: (W,)
        #   x_last: (W,)

        # Hard guards: never persist NaN/Inf artifacts
        if not np.isfinite(out["yhat"]).all():
            raise RuntimeError(f"{gen_root.name}/{run_id}: TimesFM produced non-finite yhat; aborting.")
        if not np.isfinite(out["x_last"]).all():
            raise RuntimeError(f"{gen_root.name}/{run_id}: x_last contains non-finite values; aborting.")

        # Optional: sanity-check shape against requested horizon
        if out["yhat"].ndim != 2 or out["yhat"].shape[1] != int(prediction_length):
            raise RuntimeError(
                f"{gen_root.name}/{run_id}: unexpected yhat shape {out['yhat'].shape}, "
                f"expected (W, {prediction_length})."
            )

        np.savez_compressed(pred_file, **out)

        meta = {
            "model": "google/timesfm-2.5-200m-pytorch",
            "task": "harmonic_multistep",
            "series_key": series_key,
            "window_length": window_length,
            "prediction_length": prediction_length,
            "step_size": step_size,
            "alignment": "yhat[w, h] predicts x[window_end[w] + 1 + h]",
        }
        meta_file.write_text(json.dumps(meta, indent=2, sort_keys=True))

        print(f"{gen_root.name}/{run_id}: saved {out['yhat'].shape[0]} windows")








if __name__ == "__main__":



    # TimesFM block

    print("Running TimesFM over generator trajectories...")

    timesfm_1step = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
    )

    timesfm_1step.compile(
        timesfm.ForecastConfig(
            max_context=256,      # conservative, enough for RW/AR1
            max_horizon=1,        # we only do 1-step here
            normalize_inputs=True,
            use_continuous_quantile_head=False,
            force_flip_invariance=False,
            infer_is_positive=False,
            fix_quantile_crossing=False,
        )
    )
    print("Random Walk with Drift:")

    run_timesfm_over_rw(
        timesfm_1step,
        Path("artifacts/trajectories/rw_drift"),
        series_key="x",
        target_kind="diff",
        window_length=50,
    )

    print("Random Walk completed.")

    print("AR(1):")

    run_timesfm_over_ar1(
    timesfm_1step,
    Path("artifacts/trajectories/ar1"),
    window_length=50,
    series_key="x",
    )
    print("AR(1) completed.")

    timesfm_multistep = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
    )

    timesfm_multistep.compile(
        timesfm.ForecastConfig(
            max_context=256,      # conservative, enough for RW/AR1
            max_horizon=256,        # enough for harmonic multistep
            normalize_inputs=True,
            use_continuous_quantile_head=False,
            force_flip_invariance=False,
            infer_is_positive=False,
            fix_quantile_crossing=False,
        )
    )
    print("Harmonic Oscillator:")
    run_timesfm_over_harmonic(
    timesfm_multistep,
    Path("artifacts/trajectories/harmonic"),
    window_length=100,
    prediction_length=200,
    step_size=1,
    series_key="x",
    )

    print("Harmonic Oscillator completed.")
    print("TimesFM predictions completed.")
    print("All predictions completed.")

