# CT3 – Counterfactual Twin-Trajectory Test

**CT3** is an evaluation framework for testing ability to learn causal relationships between parameters of time series generators and their outcomes.

The philosophy is simple:

> **You should be able to test whether an agent detects and responds to temporal patterns—even when the environment changes in controlled ways.**

CT3 provides:

- Reproducible generator parameter sweeps  
- Counterfactual evaluation  
- A minimal experiment interface  
- JSON summaries  
- Modular design (agents / generators / tasks / factories / CT3 executor)

---

## 1. Overview

CT3 experiments generally follow this structure:

1. A **generator** with certain parameters produces a time series (e.g., slope).
2. A **task** defines how the agent interacts with the generator.
3. An **agent** takes actions based on its policy, possibly learning a polciy during a training phase.
4. CT3 then:
   - Evaluates the frozen policy on the trained on reference parameter
   - Evaluates a set of counterfactual parameter values (μ ∈ param_grid)
   - Compares differences in policy outputs

The outcome is a structured comparison of agent behavior and performance across controlled generator changes.

---

## 2. Quickstart – Running an Experiment

From the project root:

```bash
python -m experiments.251117_pred_experiment
```

To run a different experiment, replace the part after "." with the file name. All experimental files need to be in the "experiments" folder.

This does the following:

### Training
Trains a prediction agent on a simple `LinearTrendGenerator` to learn a policy for the prediction task.

### Freeze policy
The agent’s parameters are frozen so CT3 evaluates a fixed policy.

### Build CT3 sweep
A `build_all(...)` function constructs:

- **reference tasks** at μ_ref  
- **tasks for each μ** in the grid  
- **multiple noise paths** per μ

### Run CT3
`CT3SingleExecutor` runs all tasks and computes metrics.

### Write results
Results are saved under `results/`:

results/
  └── YYYYMMDD-HHMMSS/
      └── PredictionTask/
          └── SGDPredictionAgent/
              └── LinearTrend/
                  ├── k=0/
                  │   └── metrics.json
                  ├── k=1/
                  │   └── metrics.json
                  ├── ...
                  └── config_snapshot.json


Each `metrics.json` contains metrics for one noise path `k` across the entire param grid.
