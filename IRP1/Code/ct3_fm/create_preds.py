# create_preds.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from chronos import Chronos2Pipeline
import timesfm

from models import *

def iter_runs(gen_root: Path):
    for run_dir in sorted(gen_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if (run_dir / "trajectory.npz").exists() and (run_dir / "meta.json").exists():
            yield run_dir


# Chronos over generator functions

def run_chronos_over_rw(
    pipeline,
    gen_root: Path,
    *,
    series_key: str,
    target_kind: str,
    window_length: int,
):
    for run_dir in iter_runs(gen_root):
        run_id = run_dir.name

        pred_dir = run_dir / "predictions/chronos"
        pred_dir.mkdir(exist_ok=True)

        pred_file = pred_dir / "chronos2_rolling_1step.npz"
        meta_file = pred_dir / "chronos2_rolling_1step_meta.json"


        # ✅ skip if predictions already exist
        npz_files = list(pred_dir.glob("*.npz"))
        json_files = list(pred_dir.glob("*.json"))
       
        if len(npz_files) == 1 and len(json_files) == 1:
            print(f"{gen_root.name}/{run_id}: predictions already exist, skipping...")
            continue

        x = load_npz_series(run_dir, key=series_key)
        y = make_target(x, target_kind=target_kind)

        yhat, t_idx = chronos_rolling_1step_predict(
            pipeline,
            y,
            run_id=run_id,
            window_length=window_length,
        )

        np.savez_compressed(
            pred_file,
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

        meta_file.write_text(json.dumps(meta, indent=2, sort_keys=True))

        print(f"{gen_root.name}/{run_id}: saved {len(yhat)} predictions")





def run_chronos_over_ar1(
    pipeline,
    gen_root: Path,
    *,
    window_length: int = 50,
    series_key: str = "x",
):
    for run_dir in sorted(d for d in gen_root.iterdir() if d.is_dir()):
        
        if not (run_dir / "trajectory.npz").exists():
            continue

        run_id = run_dir.name

        pred_dir = run_dir / "predictions/chronos"
        pred_dir.mkdir(exist_ok=True)

        # ✅ skip if predictions already exist
        npz_files = list(pred_dir.glob("*.npz"))
        json_files = list(pred_dir.glob("*.json"))
        if len(npz_files) == 1 and len(json_files) == 1:
            print(f"{gen_root.name}/{run_id}: predictions already exist, skipping...")
            continue


        x = load_npz_series(run_dir, key=series_key)

        yhat, t_idx, x_last = chronos_ar1_rolling_1step_predict_levels(
            pipeline,
            x,
            run_id=run_id,
            window_length=window_length,
        )


        np.savez_compressed(
            pred_dir / "chronos2_ar1_levels_rolling_1step.npz",
            yhat=yhat,
            t_idx=t_idx,
            x_last=x_last,
        )

        meta = {
            "model": "amazon/chronos-2",
            "task": "ar1_levels_rolling_1step",
            "prediction_length": 1,
            "quantile_levels": [0.5],
            "window_length": window_length,
            "series_key": series_key,
            "alignment": "yhat[i] predicts x[t_idx[i] + 1]",
        }
        (pred_dir / "chronos2_ar1_levels_rolling_1step_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True)
        )

        print(f"ar1/{run_id}: saved {len(yhat)} forecasts")


def run_chronos_over_harmonic(
    pipeline,
    gen_root: Path,
    *,
    series_key: str = "x",
    window_length: int = 100,
    prediction_length: int = 200,
    step_size: int = 1,
    quantile_levels: tuple[float, ...] = (0.5,),
):
    for run_dir in iter_runs(gen_root):
        run_id = run_dir.name

        pred_dir = run_dir / "predictions/chronos"
        pred_dir.mkdir(exist_ok=True)

        pred_file = pred_dir / "chronos2_harmonic_multistep.npz"
        meta_file = pred_dir / "chronos2_harmonic_multistep_meta.json"

        # ✅ skip if predictions already exist
        npz_files = list(pred_dir.glob("*.npz"))
        json_files = list(pred_dir.glob("*.json"))
        if len(npz_files) == 1 and len(json_files) == 1:
            print(f"{gen_root.name}/{run_id}: predictions already exist, skipping...")
            continue

        x = load_npz_series(run_dir, key=series_key)

        out = chronos_harmonic_multistep_forecast(
            pipeline,
            x,
            run_id=run_id,
            window_length=window_length,
            prediction_length=prediction_length,
            step_size=step_size,
            quantile_levels=quantile_levels,
        )

        np.savez_compressed(pred_file, **out)

        meta = {
            "model": "amazon/chronos-2",
            "task": "harmonic_multistep",
            "series_key": series_key,
            "window_length": window_length,
            "prediction_length": prediction_length,
            "step_size": step_size,
            "quantile_levels": list(quantile_levels),
            "alignment": "yhat_q[w, h, q] predicts x[window_end[w] + 1 + h]",
        }
        meta_file.write_text(json.dumps(meta, indent=2, sort_keys=True))

        print(f"{gen_root.name}/{run_id}: saved {out['yhat_q'].shape[0]} windows")


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
        pred_dir.mkdir(parents=True, exist_ok=True)

        pred_file = pred_dir / "timesfm_rolling_1step.npz"
        meta_file = pred_dir / "timesfm_rolling_1step_meta.json"

        # ✅ skip if predictions already exist (1 npz + 1 json invariant per model)
        npz_files = list(pred_dir.glob("*.npz"))
        json_files = list(pred_dir.glob("*.json"))

        if len(npz_files) == 1 and len(json_files) == 1:
            print(f"{gen_root.name}/{run_id}: predictions already exist, skipping...")
            continue

        # Safety: fail loudly if multiple artifacts exist
        if len(npz_files) > 1 or len(json_files) > 1:
            raise RuntimeError(
                f"{gen_root.name}/{run_id}: ambiguous prediction artifacts in {pred_dir} "
                f"(npz={[p.name for p in npz_files]}, json={[p.name for p in json_files]})"
            )

        x = load_npz_series(run_dir, key=series_key)
        y = make_target(x, target_kind=target_kind)

        yhat, t_idx = timesfm_rolling_1step_predict(
            model,
            y,
            window_length=window_length,
        )

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
        pred_dir.mkdir(parents=True, exist_ok=True)

        # ✅ skip if predictions already exist
        npz_files = list(pred_dir.glob("*.npz"))
        json_files = list(pred_dir.glob("*.json"))
        if len(npz_files) == 1 and len(json_files) == 1:
            print(f"{gen_root.name}/{run_id}: predictions already exist, skipping...")
            continue

        # Safety: fail loudly if multiple artifacts exist
        if len(npz_files) > 1 or len(json_files) > 1:
            raise RuntimeError(
                f"{gen_root.name}/{run_id}: ambiguous prediction artifacts in {pred_dir} "
                f"(npz={[p.name for p in npz_files]}, json={[p.name for p in json_files]})"
            )

        x = load_npz_series(run_dir, key=series_key)

        # TimesFM forecasts directly on levels for AR(1)
        yhat, t_idx = timesfm_rolling_1step_predict(
            model,
            x,
            window_length=window_length,
        )

        # last observed x_t in each rolling window (aligns with t_idx)
        # window covers [start_idx, end_idx-1] where end_idx = start_idx + window_length
        # t_idx is end_idx - 1, so x_last is x[t_idx]
        x_last = x[t_idx].astype(np.float32, copy=False)

        np.savez_compressed(
            pred_dir / "timesfm_ar1_levels_rolling_1step.npz",
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
        (pred_dir / "timesfm_ar1_levels_rolling_1step_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True)
        )

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
        pred_dir.mkdir(parents=True, exist_ok=True)

        pred_file = pred_dir / "timesfm_harmonic_multistep.npz"
        meta_file = pred_dir / "timesfm_harmonic_multistep_meta.json"

        # ✅ skip if predictions already exist
        npz_files = list(pred_dir.glob("*.npz"))
        json_files = list(pred_dir.glob("*.json"))
        if len(npz_files) == 1 and len(json_files) == 1:
            print(f"{gen_root.name}/{run_id}: predictions already exist, skipping...")
            continue

        # Safety: fail loudly if multiple artifacts exist
        if len(npz_files) > 1 or len(json_files) > 1:
            raise RuntimeError(
                f"{gen_root.name}/{run_id}: ambiguous prediction artifacts in {pred_dir} "
                f"(npz={[p.name for p in npz_files]}, json={[p.name for p in json_files]})"
            )

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

    # Chronos block

    print("Running Chronos over generator trajectories...")

    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
    )

    run_chronos_over_rw(
        pipeline,
        Path("artifacts/trajectories/rw_drift"),
        series_key="x",
        target_kind="diff",
        window_length=50,
    )

    run_chronos_over_ar1(
    pipeline,
    Path("artifacts/trajectories/ar1"),
    window_length=50,
    series_key="x",
    )

    run_chronos_over_harmonic(
    pipeline,
    Path("artifacts/trajectories/harmonic"),
    window_length=100,
    prediction_length=200,
    step_size=1,
    series_key="x",
    quantile_levels=(0.5,),
    )

    print("Chronos predictions completed.")


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

    # Moirai block
