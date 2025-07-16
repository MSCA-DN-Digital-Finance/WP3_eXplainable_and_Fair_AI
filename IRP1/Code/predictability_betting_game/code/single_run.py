import argparse
import sys
import json
from pathlib import Path

from agent_factory import create_agent
from generator_factory import create_generator
from tsdm.games import BettingGame


def parse_args(argv=None):
    p = argparse.ArgumentParser("Run a single agent-generator betting game instance")
    p.add_argument("--experiment", required=True, help="Path to results folder provided by orchestrator")
    p.add_argument("--generator", required=True, help="Generator name")
    p.add_argument("--agent", required=True, help="Agent name")
    p.add_argument("--run", required=True, help="Run ID (e.g., 01)")
    p.add_argument("--steps", type=int, default=10_000, help="Total movements/steps in game")
    p.add_argument("--start_value", type=float, default=0.0, help="Starting value of the game")
    p.add_argument("--agent_params", type=str, default="{}", help="JSON string of agent params")
    p.add_argument("--generator_params", type=str, default="{}", help="JSON string of generator params")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    try:
        agent_params = json.loads(args.agent_params)
        generator_params = json.loads(args.generator_params)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Parameter parsing error: {e}\n")
        return 2

    agent = create_agent(args.agent, agent_params)
    generator = create_generator(args.generator, generator_params)

    game = BettingGame(generator, agent, total_movements=args.steps, start_value=args.start_value)
    final_reward = game.play_game()

    results = {
        "experiment_results_path": str(args.experiment),
        "agent": args.agent,
        "agent_params": agent_params,
        "generator": args.generator,
        "generator_params": generator_params,
        "run_id": args.run,
        "total_steps": args.steps,
        "start_value": args.start_value,
        "final_cumulative_reward": final_reward,
        "reward_development": game.reward_development.tolist(),
        "bet_log": [
            {"step": step, "value": value, "bet": bet, "reward": reward}
            for step, value, bet, reward in game.bet_log
        ]
    }

    # Save to <experiment_path>/<agent>/<generator>/<run>.json
    out_dir = Path(args.experiment) / args.agent / args.generator
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
