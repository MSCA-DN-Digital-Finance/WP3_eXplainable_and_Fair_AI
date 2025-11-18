"""
================================================================================
Experiment: PredictionTask with LinearTrend Generator (CT3 Evaluation)
================================================================================

Goal:
    Evaluate how an SGD-based prediction agent responds to linear trends by
    sweeping the slope μ and measuring metrics under CT3.

Overview:
    1) Warm-up (optional): train the agent on a simple LinearTrendGenerator
       (e.g., slope = 0.2) to train a policy.

    2) Freeze the agent to keep policy fixed during evaluation.

    3) CT3 evaluation:
         - Generator: LinearTrend(slope = μ)
           with μ ∈ [-0.5, -0.25, 0.0, 0.25, 0.5]
         - NoiseTrajectoryWrapper introduces stochasticity for robustness.
         - Metric: Prediction metrics.

Key parameters:
    - task_horizon: 1000 steps per rollout
    - n_noise_paths: 8 (independent noise draws per μ)
    - seed: 123
    - out_root: "results"

Expected outcome:
    A summary JSON under "results/" showing how prediction metrics vary with μ.
================================================================================
"""

from ct3_single_run import CT3Config
from orchestrator import run_suite
from custom_generators import NoiseTrajectoryWrapper
from tsg.generators import LinearTrendGenerator
from tsg.modifiers  import GaussianNoise
from custom_agents import SGDClassifierAgent  # <-- use your prediction SGD agent class
from factories import  make_pred_task_factory
from tsdm.tasks import PredictionTask  # <-- prediction task
from metrics import metric_prediction         # <-- prediction metric

# ---- Agent ----
pred_agent = SGDClassifierAgent(random_state=42)

# ---- Optional warm-up before CT3 ----

ref_param = 0.2

print("Defining training generator...")
lin_gen = LinearTrendGenerator(slope=ref_param)
train_gen = GaussianNoise(lin_gen, mu=0, sigma=0.5)

print("Training agent on generator...")
task = PredictionTask(
    generator=train_gen,
    agent=pred_agent,
    total_movements=1000,         
)
_ = task.play_game()
print("Task Log (first 5):", task.log[:5])

print("Training done. Freezing policy...")
# If your SGDAgent has these; otherwise remove/adjust
if hasattr(pred_agent, "freeze"):
    pred_agent.freeze()
if hasattr(pred_agent, "soft_reset"):
    pred_agent.soft_reset()

# ---- CT3: build task factory with a single linear-trend generator (slope=μ) ----
print("Building CT3 task factory...")
trend_factory = lambda mu: LinearTrendGenerator(slope=mu)

build_all_pred = make_pred_task_factory(
    task_cls=PredictionTask,
    gen_factory=trend_factory,           # single generator for prediction
    agent=pred_agent,
    noise_wrapper_cls=NoiseTrajectoryWrapper,
    # pass your task kwargs that factory expects:
    total_movements=1000,                        
)

# ---- CT3 config ----
cfg = CT3Config(
    task_name="PredictionTask",
    agent_name="SGDAgent",
    generator_name="LinearTrend",
    build_all=build_all_pred,
    param_grid=[-0.5, -0.25, 0.0, 0.25, 0.5],
    reference_param=ref_param,                   # baseline slope (for ref rollouts)
    task_horizon=1000,
    n_noise_paths=8,
    seed=123,
    metrics=[metric_prediction()],                # prediction metrics
    out_root="results",
)

print("Running CT3...")
summary = run_suite([cfg])
print(summary["summary_path"])
print("Experiment completed.")
