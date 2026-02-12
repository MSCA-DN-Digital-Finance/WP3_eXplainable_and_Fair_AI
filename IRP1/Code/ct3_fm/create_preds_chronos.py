# create_preds.py
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from chronos import Chronos2Pipeline

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
        pred_dir.mkdir(parents=True, exist_ok=True)

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
        pred_dir.mkdir(parents=True, exist_ok=True)

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
        pred_dir.mkdir(parents=True, exist_ok=True)

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


def run_chronos_over_solar(
    pipeline,
    file: Path,
    *,
    series_key: str = "x",
    window_length: int = 100,
    prediction_length: int = 200,
    step_size: int = 1,
    quantile_levels: tuple[float, ...] = (0.5,),
):
    
    print(f"Processing solar data from {file}...")
    summer_traj = np.load(file)['summer']
    winter = np.load(file)['winter']

    print(f"Running Chronos multistep forecast on solar summer trajectory...")

    summer_out = chronos_harmonic_multistep_forecast(
        pipeline,
        summer_traj,
        run_id="summer",
        window_length=window_length,
        prediction_length=prediction_length,
        step_size=step_size,
        quantile_levels=quantile_levels,
    )

    sum_pred_file = Path(file).parent / "predictions" / "chronos" / "chronos2_solar_summer.npz"

    os.makedirs(os.path.dirname(sum_pred_file), exist_ok=True)
    np.savez_compressed(sum_pred_file, **summer_out)

    summer_meta = {
        "model": "amazon/chronos-2",
        "task": "harmonic_multistep",
        "series_key": series_key,
        "window_length": window_length,
        "prediction_length": prediction_length,
        "step_size": step_size,
        "quantile_levels": list(quantile_levels),
        "alignment": "yhat_q[w, h, q] predicts x[window_end[w] + 1 + h]",
    }
    summer_meta_file = Path(file).parent / "predictions" / "chronos" / "chronos2_solar_summer_meta.json"
    summer_meta_file.write_text(json.dumps(summer_meta, indent=2, sort_keys=True))

    print(f"Running Chronos multistep forecast on solar winter trajectory...")

    winter_out = chronos_harmonic_multistep_forecast(
        pipeline,
        winter,
        run_id="winter",
        window_length=window_length,
        prediction_length=prediction_length,
        step_size=step_size,
        quantile_levels=quantile_levels,
    )
    win_pred_file = Path(file).parent / "predictions" / "chronos" / "chronos2_solar_winter.npz"
    os.makedirs(os.path.dirname(win_pred_file), exist_ok=True)
    np.savez_compressed(win_pred_file, **winter_out)

    winter_meta = {
        "model": "amazon/chronos-2",
        "task": "harmonic_multistep",
        "series_key": series_key,
        "window_length": window_length,
        "prediction_length": prediction_length,
        "step_size": step_size,
        "quantile_levels": list(quantile_levels),
        "alignment": "yhat_q[w, h, q] predicts x[window_end[w] + 1 + h]",
    }
    win_meta_file = Path(file).parent / "predictions" / "chronos" / "chronos2_solar_winter_meta.json"
    win_meta_file.write_text(json.dumps(winter_meta, indent=2, sort_keys=True))

    print(f"Chronos predictions for solar data completed. Saved to {sum_pred_file} and {win_pred_file}.")


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

    run_chronos_over_solar(
        pipeline,
        Path("data/solar_summer_winter.npz"),
        series_key="x",
        window_length=100,
        prediction_length=200,
        step_size=1,
        quantile_levels=(0.5,),
    )   

    print("Chronos predictions completed.")

