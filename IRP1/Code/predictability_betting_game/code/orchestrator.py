import subprocess
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from aggregation_functions import aggregate_results



RUN_CONFIG = {
    "steps_per_run": 100,
    "runs_per_combination": 2,
    "start_value": 0.0,
    "max_workers": 8
}


AGENT_CONFIG = {
    # Zero-Intelligence Class
    "Always Up Agent": {},

    # Fixed-Rule Class
    "Repeat Last Movement Agent": {},

    # Stationary Statistical Class
    "Frequency-Based Majority Agent": {},
    "Static Mean Reversion Agent": {},

    # Dynamic Statistical Class
    "Dynamic Mean Reversion Agent": {
        "time_window": 10 
    },

    # Inferring Model Class
    "SGD Classifier Agent": {
        "window_size": 50
    },
    "DQN Agent": {
        "state_size": 10,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "gamma": 0.95,
        "lr": 1e-3,
        "batch_size": 32,
        "memory_size": 1000
    }
}

GENERATOR_CONFIG = {
    "Real World Asset Price Generator": {
        "ticker": "AAPL",
        "start": "1985-01-01",
        "end": "2025-01-01"
    },
  "Linear Trend Generator": {
    "start_value": 10.0,
    "slope": 0.5
  },
  "Constant Generator": {},
  "Periodic Trend Generator": {
    "start_value": 10.0,
    "amplitude": 2.0,
    "frequency": 0.1
  },
  "Ornstein-Uhlenbeck Generator": {
    "mu": 10.0,
    "theta": 0.15,
    "sigma": 0.2,
    "dt": 1.0,
    "start_value": 20.0
  },
  "Random Walk Generator": {
    "start_value": 0.0,
    "mu": 0.0,
    "sigma": 1.0
  },
  "Markov Regime-Switching Generator": {
    "regimes": [
      {
        "generator": "Linear Trend Generator",
        "params": {
          "start_value": 0.0,
          "slope": 0.5
        }
      },
      {
        "generator": "Random Walk Generator",
        "params": {
          "start_value": 0.0,
          "mu": 0.0,
          "sigma": 1.0
        }
      }
    ],
    "transition_matrix": [
      [0.8, 0.2],
      [0.2, 0.8]
    ]
  },
  "Noisy Markov Regime-Switching Generator": {
    "regimes": [
      {
        "generator": "Linear Trend Generator",
        "params": {
          "start_value": 0.0,
          "slope": 0.5
        }
      },
      {
        "generator": "Ornstein-Uhlenbeck Generator",
        "params": {
          "mu": 0.0,
          "theta": 0.1,
          "sigma": 0.1,
          "dt": 1.0,
          "start_value": 0.0
        }
      }
    ],
    "transition_matrix": [
      [0.8, 0.2],
      [0.2, 0.8]
    ],
    "noise_mu": 0.0,
    "noise_sigma": 0.1
  }
}



DETERMINISTIC_AGENTS = {} #{"Always Up Agent", "Repeat Last Movement Agent", "Static Mean Reversion Agent"}
DETERMINISTIC_GENERATORS = {} #{"Linear Trend Generator", "Constant Generator", "Periodic Trend Generator"}


def get_logger(log_file_path):
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        with open(log_file_path, "a") as f:
            f.write(formatted + "\n")
    return log


def should_skip(agent_name, generator_name):
    return agent_name in DETERMINISTIC_AGENTS and generator_name in DETERMINISTIC_GENERATORS


def run_single(agent_name, agent_params, generator_name, generator_params, run_id, experiment_name, results_dir, steps, start_value, log_file):
    log = get_logger(log_file)
    cmd = [
        "python", "single_run.py",
        "--experiment", str(results_dir),  # Pass results folder to single_run
        "--generator", generator_name,
        "--agent", agent_name,
        "--run", f"{run_id:03d}",
        "--steps", str(steps),
        "--start_value", str(start_value),
        "--agent_params", json.dumps(agent_params),
        "--generator_params", json.dumps(generator_params)
    ]
    log(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        log(f"Run completed: {agent_name} x {generator_name} | Run {run_id:03d}")
    except subprocess.CalledProcessError as e:
        log(f"Run failed: {e}")


def orchestrate_all_runs():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"exp_{timestamp}"
    experiment_dir = Path("../data/experiment") / experiment_name
    results_dir = experiment_dir / "results"
    log_file = experiment_dir / f"{experiment_name}.log"

    logger = get_logger(log_file)
    logger(f"Starting experiment: {experiment_name}")

    steps = RUN_CONFIG["steps_per_run"]
    runs_per_combination = RUN_CONFIG["runs_per_combination"]
    start_value = RUN_CONFIG["start_value"]
    max_workers = RUN_CONFIG["max_workers"]

    results_dir.mkdir(parents=True, exist_ok=True)

    config_out = experiment_dir / "experiment_config.json"
    with open(config_out, "w") as f:
        json.dump({
            "run_config": RUN_CONFIG,
            "agent_config": AGENT_CONFIG,
            "generator_config": GENERATOR_CONFIG,
            "deterministic_agents": list(DETERMINISTIC_AGENTS),
            "deterministic_generators": list(DETERMINISTIC_GENERATORS),
            "experiment_name": experiment_name,
            "timestamp": timestamp
        }, f, indent=2)
    logger(f"Saved experiment config to {config_out}")

    tasks = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for agent_name, agent_params in AGENT_CONFIG.items():
            for generator_name, generator_params in GENERATOR_CONFIG.items():
                if should_skip(agent_name, generator_name):
                    logger(f"Skipping deterministic pair: {agent_name} x {generator_name}")
                    continue
                for run_id in range(1, runs_per_combination + 1):
                    tasks.append(executor.submit(
                        run_single,
                        agent_name, agent_params,
                        generator_name, generator_params,
                        run_id, experiment_name, results_dir, steps, start_value, log_file
                    ))

        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                logger(f"Error during run execution: {e}")

    logger(f"All runs completed. Aggregating results...")
    
    summary_out = experiment_dir / "summary_per_run.csv"
    convergence_out = experiment_dir / "convergence_per_step.csv"

    aggregate_results(results_dir, summary_out, convergence_out)

    logger(f"Aggregation completed. Summary and convergence data saved.")

    logger(f"Experiment {experiment_name} completed.")


if __name__ == "__main__":
    orchestrate_all_runs()
