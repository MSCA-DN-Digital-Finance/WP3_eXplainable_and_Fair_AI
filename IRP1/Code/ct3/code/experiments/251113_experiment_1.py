"""
================================================================================
Experiment: AllocationTask with Const + Trend Generators (CT3 Evaluation)
================================================================================

Goal:
    Evaluate how an allocation agent (SGDAllocAgent) responds to a known 
    directional trend under controlled generator conditions using the CT3 
    benchmarking framework.

Overview:
    1. The agent is first "warmed up" (trained) on a simple environment with 
       two LinearTrendGenerators — one constant (slope = 0.0) and one upward 
       trend (slope = 0.2). This pretraining phase helps the agent learn a 
       baseline allocation behavior.

    2. After training, the agent is frozen (policy fixed) to ensure consistent 
       evaluation. Its parameters are not updated during CT3 runs.

    3. For evaluation, a task factory is created using:
         - Generator 1: Constant trend (slope = 0.0)
         - Generator 2: Variable trend (slope = μ)
       where μ ∈ [-0.5, -0.25, 0.0, 0.25, 0.5].
       This defines a sweep of different trend intensities, both negative and 
       positive, to test whether the agent's allocations respond appropriately 
       to directional changes.

    4. Each environment is wrapped with NoiseTrajectoryWrapper to introduce 
       stochasticity and evaluate robustness.

    5. The CT3 framework runs these task variants, logs agent performance, and 
       computes the allocation metrics (alloc_L1, alloc_response) via 
       metric_allocation().

Key parameters:
    - task_horizon: 1000 movements per trajectory
    - n_noise_paths: 8 stochastic samples per μ
    - tc: transaction cost (here set to 0.0 for simplicity)
    - out_root: "experiments" (output directory for logs and summaries)
    - seed: 123 (for reproducibility)

Expected outcome:
    The final summary report (JSON) under "experiments/" will show how the 
    agent's allocation magnitude and directionality change with increasing or 
    decreasing trend slopes. A well-behaved agent should allocate more towards 
    positively trending assets and less towards negatively trending ones.

================================================================================
"""


from ct3_single_run import CT3Config
from orchestrator import run_suite
from custom_generators import NoiseTrajectoryWrapper
from tsg.generators import LinearTrendGenerator
from custom_agents import SGDAllocAgent
from factories import make_task_factory
from tsdm.tasks import AllocationTask
from metrics import metric_allocation

alloc_agent = SGDAllocAgent(n_assets=2, random_state=42)
gens = [LinearTrendGenerator(slope=0.0), LinearTrendGenerator(slope=0.2)]
# train the agent a bit before CT3 runs
print("Training agent...")
task = AllocationTask(
    generators=gens,
    agent=alloc_agent,
    total_movements=1000,
    start_values=[0.0, 0.0],
    tc=0.0
    )

_ = task.play_game()
print("Task Log:", task.log)
print("Training done.")

print("Freezing policy...")
alloc_agent.freeze(); alloc_agent.soft_reset()


# evaluate on CT3
print("Building CT3 task factory...")
const_factory  = lambda mu: LinearTrendGenerator(slope=0.0)
trend_factory  = lambda mu: LinearTrendGenerator(slope=mu)
build_all_alloc = make_task_factory(
    task_cls=AllocationTask,
    gen_factory=[const_factory, trend_factory],
    agent=alloc_agent,
    noise_wrapper_cls=NoiseTrajectoryWrapper,
    total_movements=1000,
    start_values=[0.0, 0.0],
    tc=0.0,
)

cfg = CT3Config(
    task_name="AllocationTask",
    agent_name="SGDAllocAgent",
    generator_name="Const+Trend",
    build_all=build_all_alloc,
    param_grid=[-0.5, -0.25, 0.0, 0.25, 0.5],
    reference_param=0.1,
    task_horizon=1000,
    n_noise_paths=8,
    seed=123,
    metrics=[metric_allocation(trend_idx=1, warmup=0)],
    out_root="results",
)
print("Running CT3...")
summary = run_suite([cfg])
print(summary["summary_path"])

print("Experiment completed.")
