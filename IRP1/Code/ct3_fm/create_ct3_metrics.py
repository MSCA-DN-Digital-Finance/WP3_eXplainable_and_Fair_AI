# run_metrics.py
from __future__ import annotations

from pathlib import Path

from ct3 import run_ct3_and_save


def main() -> None:
    # rw_drift: intervention on mu
    run_ct3_and_save(
        Path("artifacts/trajectories/rw_drift"),
        intervention_key="mu",
        out_path=Path("artifacts/ct3/rw_drift_mu.csv"),
        metric_name="prob_pos_forecast",
    )


    print("CT3 metrics computed and saved.")


if __name__ == "__main__":
    main()
