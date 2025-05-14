# base_trainer.py – parameterised single run
# -----------------------------------------------------------------------------
"""Train one (experiment, trend, agent, run) episode and save logs/plots.

Called by orchestrator like:
    python base_trainer.py --experiment 4_1 --trend upward --agent ddpg --run 01

Outputs are written under:
    Results/<experiment>/<agent>/<trend>/{training_logs,plots}/

Exit status 0  → run finished OK
Exit status >0 → run crashed; orchestrator will mark FAILED
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent
from environments import StockPortfolioEnv
from plot_functions import (
    preprocess_df,
    plot_dual_line_per_episode,
    plot_per_episode_row,
)

# ---------------------------------------------------------------------------
# Constants / hyper‑params
# ---------------------------------------------------------------------------

def get_hyper(agent: str) -> dict:
    table = {
        "ddpg": {
            "batch_size": 128,
            "buffer_size": 50000,
            "learning_rate": 0.001,
        },
        "sac": {
            "batch_size": 128,
            "buffer_size": 100000,
            "learning_rate": 0.0003,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
        },
        "a2c": {
            "n_steps": 5,
            "ent_coef": 0.005,
            "learning_rate": 0.0002,
        }
    }
    return table[agent]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser("Train a single RL run")
    p.add_argument("--experiment", required=True, help="4_1 or 4_2")
    p.add_argument("--trend", required=True, help="trend name e.g. upward_noise")
    p.add_argument("--agent", required=True, choices=["ddpg", "sac", "a2c"], help="agent type")
    p.add_argument("--run", required=True, help="run id, e.g. 01")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--days", type=int, default=1000)
    return p.parse_args(argv)

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main(argv=None):  # noqa: C901
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # Data load
    # ------------------------------------------------------------------
    csv_path = Path("Data") / f"{args.trend}.csv"
    if not csv_path.exists():
        sys.stderr.write(f"CSV not found: {csv_path}\n")
        return 2  # non‑zero to signal error

    train = pd.read_csv(csv_path)

    stock_dim = train["tic"].nunique()
    state_space = len(["close"]) * stock_dim

    env_kwargs = {
        "initial_amount": 1000,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "action_space": stock_dim,
    }

    env = StockPortfolioEnv(df=train, **env_kwargs)
    vec_env, _ = env.get_sb_env()

    # ------------------------------------------------------------------
    # Agent setup & training
    # ------------------------------------------------------------------
    agent = DRLAgent(env=vec_env)
    model = agent.get_model(args.agent, model_kwargs=get_hyper(args.agent))

    total_steps = args.episodes * args.days
    agent.train_model(model=model, tb_log_name=args.agent, total_timesteps=total_steps)

    # ------------------------------------------------------------------
    # create directories
    # ------------------------------------------------------------------
    base_dir  = Path("Results") / args.experiment / args.agent / args.trend
    logs_dir  = base_dir / "training_logs"
    plots_dir = base_dir / "plots"            # root for all runs
    run_dir   = plots_dir / args.run          # e.g. …/plots/03/

    for d in (logs_dir, run_dir):
        d.mkdir(parents=True, exist_ok=True)

    csv_out = logs_dir / f"{args.run}.csv"
    env.save_episode_log(csv_out)

    # ------------------------------------------------------------------
    # Post‑processing & plots
    # ------------------------------------------------------------------
    df_logs = preprocess_df(pd.read_csv(csv_out))

    plot_per_episode_row(
        df_logs, "new_portfolio_value", "Portfolio Value ($)",
        "Portfolio Value Across Episodes",
        run_dir / "portfolio_value.png",      # <- inside the run folder
    )

    plot_per_episode_row(
        df_logs, "portfolio_return", "Portfolio Return (%)",
        "Portfolio Return Across Episodes",
        run_dir / "portfolio_return.png",
    )

    plot_per_episode_row(
        df_logs, "reward", "Reward",
        "Reward Across Episodes",
        run_dir / "reward.png",
    )

    plot_dual_line_per_episode(
        df_logs, "actions", "actions",
        "Allocation Weights Before Normalization",
        "Allocation Weights Before Normalization Over Time",
        run_dir / "actions.png",
        legend1="Action 1", legend2="Action 2",
    )

    plot_dual_line_per_episode(
        df_logs, "allocation_weights", "allocation_weights",
        "Allocation Weights", "Allocation Over Time",
        run_dir / "allocation.png",
        legend1="Cash Weight", legend2="Stock Weight",
    )


    return 0


if __name__ == "__main__":
    sys.exit(main())