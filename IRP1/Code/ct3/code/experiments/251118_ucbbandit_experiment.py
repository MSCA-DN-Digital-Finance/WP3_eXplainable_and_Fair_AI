"""
================================================================================
Experiment: Directional Bandit on PredictionTask with LinearTrend Generator (CT3)
================================================================================

Goal:
    Evaluate how a non-contextual UCB bandit agent performs on a directional
    prediction task when the underlying time series exhibits different linear
    trend slopes μ.

    The agent has two actions:
        0 = "down", 1 = "up"
    and receives reward +1 if it correctly predicts the direction of the next
    movement (sign of y_t - y_{t-1}), and -1 otherwise.

Overview:
    1) Warm-up:
         Train the UCBBanditAgent on a reference LinearTrendGenerator
         (e.g., slope = 0.2) so that it builds initial estimates of which
         action ("up" vs "down") is more profitable in that regime.

    2) Freeze the agent:
         After warm-up, the bandit is frozen so that its statistics (means,
         counts) are not updated during CT3 evaluation. We only evaluate a
         fixed policy.

    3) CT3 evaluation:
         - Generator: LinearTrendGenerator(slope = μ)
           with μ ∈ [-0.5, -0.25, 0.0, 0.25, 0.5]
         - Optional: NoiseTrajectoryWrapper can be used to add stochasticity
           for robustness tests (can be disabled for initial experiments).
         - Task: directional PredictionTask where the agent predicts the
           sign of the next movement using actions {down, up}.
         - Metrics: directional prediction metrics (e.g. accuracy or mean
           reward) computed from the task log.

Key parameters:
    - task_horizon: 1000 steps per rollout
    - n_noise_paths: 8 (independent noise draws per μ)
    - seed: 123
    - out_root: "results"

Expected outcome:
    A summary JSON under "results/" that reports, for each slope μ, how the
    bandit's directional prediction performance (e.g. accuracy, mean reward)
    behaves across the param grid. This allows checking whether the bandit
    policy learned at the reference slope generalizes or degrades as the
    trend steepens, flattens, or changes sign.
================================================================================
"""

from ct3_single_run import CT3Config
from orchestrator import run_suite
from custom_generators import NoiseTrajectoryWrapper
from tsg.generators import LinearTrendGenerator
from tsg.modifiers import GaussianNoise
from custom_agents import UCBBanditAgent          # <-- non-contextual UCB bandit
from factories import make_pred_task_factory
from tsdm.tasks import PredictionTask             # or a directional variant if you have one
from metrics import metric_prediction
import json            
from pathlib import Path

# ---- Agent ----
bandit_agent = UCBBanditAgent(exploration_c=1.0)

# ---- Warm-up before CT3 (reference regime) ----

ref_param = 0.2

print("Defining training generator...")
lin_gen = LinearTrendGenerator(slope=ref_param)
train_gen = GaussianNoise(lin_gen, mu=0.0, sigma=0.5)  # can simplify/remove noise if desired

print("Training bandit agent on generator (reference slope)...")
task = PredictionTask(
    generator=train_gen,
    agent=bandit_agent,
    total_movements=1000,
)
_ = task.play_game()
print("Task Log (first 5):", task.log[:5])

print("Training done. Freezing bandit policy...")
if hasattr(bandit_agent, "freeze"):
    bandit_agent.freeze()
if hasattr(bandit_agent, "soft_reset"):
    bandit_agent.soft_reset()

# ---- CT3: build task factory with a single linear-trend generator (slope = μ) ----
print("Building CT3 task factory...")
trend_factory = lambda mu: LinearTrendGenerator(slope=mu)

build_all_pred = make_pred_task_factory(
    task_cls=PredictionTask,
    gen_factory=trend_factory,           # single generator for prediction/bandit
    agent=bandit_agent,
    noise_wrapper_cls=NoiseTrajectoryWrapper,
    total_movements=1000,
)

# ---- CT3 config ----
cfg = CT3Config(
    task_name="PredictionTask",  # name for folder structure / identification
    agent_name="UCBBanditAgent",
    generator_name="LinearTrend",
    build_all=build_all_pred,
    param_grid=[-0.5, -0.25, 0.0, 0.25, 0.5],
    reference_param=ref_param,             # baseline slope (for reference rollouts)
    task_horizon=1000,
    n_noise_paths=8,
    seed=123,
    metrics=[metric_prediction()],         # directional prediction metric(s)
    out_root="results",
)

print("Running CT3...")
summary = run_suite([cfg])
print(summary["summary_path"])

# Add task log from training phase to results
summary_file = Path(summary["summary_path"])
experimment_root = summary_file.parent
train_log_path = experimment_root / "training_task_log.json"
with train_log_path.open("w", encoding="utf-8") as f:
        json.dump(task.log, f, indent=2)

print("Experiment completed.")
