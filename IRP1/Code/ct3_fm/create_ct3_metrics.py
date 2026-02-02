# run_metrics.py
from __future__ import annotations

from pathlib import Path

from ct3 import run_rw_ct3_and_save, run_ar1_ct3_and_save, run_harmonic_ct3_and_save


def main() -> None:
    from pathlib import Path

    # ----------------------------
    # Chronos CT3 outputs
    # ----------------------------
    print("Computing CT3 metrics...")
    print("Chronos CT3 outputs...")

    run_rw_ct3_and_save(
        Path("artifacts/trajectories/rw_drift/"),
        exp_id=1,
        intervention_key="mu",
        model_name="chronos",
        out_path=Path("artifacts/ct3/chronos/rw_drift_exp_1.csv"),
        metric_name="prob_pos_forecast",
    )

    run_ar1_ct3_and_save(
        Path("artifacts/trajectories/ar1"),
        exp_id=2,
        intervention_key="phi",
        model_name="chronos",
        out_path=Path("artifacts/ct3/chronos/ar1_exp_2.csv"),
        metric_name="beta_hat_model",
    )

    run_harmonic_ct3_and_save(
        Path("artifacts/trajectories/harmonic"),
        exp_id=3,
        intervention_key="omega",
        model_name="chronos",
        out_path=Path("artifacts/ct3/chronos/harmonic_exp_3.csv"),
        quantile=0.5,
        dt=1.0,
    )


    run_ar1_ct3_and_save(
            Path("artifacts/trajectories/ar1"),
            exp_id=4,
            intervention_key="sigma",
            model_name="chronos",
            out_path=Path("artifacts/ct3/chronos/ar1_exp_4.csv"),
            metric_name="beta_hat_model",
        )
    
    run_harmonic_ct3_and_save(
        Path("artifacts/trajectories/harmonic"),
        exp_id=5,
        intervention_key="A",
        model_name="chronos",
        out_path=Path("artifacts/ct3/chronos/harmonic_exp_5.csv"),
        quantile=0.5,
        dt=1.0,
    )

    print("Chronos CT3 metrics computed and saved.")
    print("TimesFM CT3 outputs...")


    # ----------------------------
    # TimesFM CT3 outputs
    # ----------------------------
    run_rw_ct3_and_save(
        Path("artifacts/trajectories/rw_drift/"),
        exp_id=1,
        intervention_key="mu",
        model_name="timesfm",
        out_path=Path("artifacts/ct3/timesfm/rw_drift_exp_1.csv"),
        metric_name="prob_pos_forecast",
    )

    run_ar1_ct3_and_save(
        Path("artifacts/trajectories/ar1"),
        exp_id=2,
        intervention_key="phi",
        model_name="timesfm",
        out_path=Path("artifacts/ct3/timesfm/ar1_exp_2.csv"),
        metric_name="beta_hat_model",
    )

    run_harmonic_ct3_and_save(
        Path("artifacts/trajectories/harmonic"),
        exp_id=3,
        intervention_key="omega",
        model_name="timesfm",
        out_path=Path("artifacts/ct3/timesfm/harmonic_exp_3.csv"),
        quantile=0.5,
        dt=1.0,
    )

    run_ar1_ct3_and_save(
        Path("artifacts/trajectories/ar1"),
        exp_id=4,
        intervention_key="sigma",
        model_name="timesfm",
        out_path=Path("artifacts/ct3/timesfm/ar1_exp_4.csv"),
        metric_name="beta_hat_model",
    )

    run_harmonic_ct3_and_save(
        Path("artifacts/trajectories/harmonic"),
        exp_id=5,
        intervention_key="A",
        model_name="timesfm",
        out_path=Path("artifacts/ct3/timesfm/harmonic_exp_5.csv"),
        quantile=0.5,
        dt=1.0,
    )

    print("TimesFM CT3 metrics computed and saved.")

    print("All CT3 metrics computed and saved.")


if __name__ == "__main__":
    main()
