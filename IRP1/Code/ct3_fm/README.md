# CT3 Experimental Codebase – Workflow Overview

This repository implements **CT3 (Counterfactual Twin-Trajectory Test)** experiments for probing **model sensitivity** of time-series foundation models (Chronos-2) under controlled generator interventions.

The pipeline is **hash-addressed, restartable, and modular**, with a strict separation between:
- data generation
- model inference
- metric computation (CT3)

---

## High-level workflow

generators.py
↓
create_data.py
↓
artifacts/trajectories/<generator>/<run_hash>/
↓
create_preds.py (Chronos inference)
↓
artifacts/trajectories/<generator>/<run_hash>/predictions/
↓
create_metrics.py (CT3 metrics)
↓
artifacts/ct3/*.csv


Each stage can be re-run independently without recomputing previous stages.



How to get started:

- install ct3-core conda env
- open vs code with code .
- in powershell execute ./run_all or .run_all -Setup to install other conda envs for chronos and timesfm inference


