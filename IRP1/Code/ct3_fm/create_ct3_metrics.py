# run_metrics.py
from __future__ import annotations

from pathlib import Path

from ct3 import run_rw_ct3_and_save, run_ar1_ct3_and_save, run_harmonic_ct3_and_save


def main() -> None:
    # rw_drift: intervention on mu
    run_rw_ct3_and_save(
        Path("artifacts/trajectories/rw_drift"),
        intervention_key="mu",
        out_path=Path("artifacts/ct3/rw_drift_mu.csv"),
        metric_name="prob_pos_forecast",
    )

    run_ar1_ct3_and_save(
    Path("artifacts/trajectories/ar1"),
    intervention_key="phi",
    out_path=Path("artifacts/ct3/ar1_phi.csv"),
    metric_name="beta_hat_model",
    )

    run_harmonic_ct3_and_save(
    Path("artifacts/trajectories/harmonic"),
    intervention_key="omega",
    out_path=Path("artifacts/ct3/harmonic_omega_w1.csv"),
    quantile=0.5,
    dt=1.0,
    )




    print("CT3 metrics computed and saved.")


if __name__ == "__main__":
    main()
