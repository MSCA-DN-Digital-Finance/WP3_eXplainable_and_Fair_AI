import json
from pathlib import Path
import pandas as pd


def collect_per_run_summary(results_dir):
    records = []
    for json_file in Path(results_dir).rglob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            records.append({
                "agent": data.get("agent"),
                "generator": data.get("generator"),
                "run_id": data.get("run_id"),
                "total_steps": data.get("total_steps"),
                "start_value": data.get("start_value"),
                "final_cumulative_reward": data.get("final_cumulative_reward"),
                "average_reward": data.get("final_cumulative_reward") / data.get("total_steps")
            })

        except Exception as e:
            print(f"[Aggregator] Failed to parse {json_file}: {e}")

    return pd.DataFrame(records)


def collect_per_step_convergence(results_dir):
    records = []
    for json_file in Path(results_dir).rglob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            for step, cumulative_reward in enumerate(data.get("reward_development", []), start=1):
                records.append({
                    "agent": data.get("agent"),
                    "generator": data.get("generator"),
                    "run_id": data.get("run_id"),
                    "step": step,
                    "cumulative_reward": cumulative_reward
                })

        except Exception as e:
            print(f"[Aggregator] Failed to parse {json_file}: {e}")

    return pd.DataFrame(records)


def aggregate_results(results_dir, summary_out_path, convergence_out_path):
    summary_df = collect_per_run_summary(results_dir)
    convergence_df = collect_per_step_convergence(results_dir)

    summary_df.to_csv(summary_out_path, index=False)
    convergence_df.to_csv(convergence_out_path, index=False)

    print(f"[Aggregator] Saved per-run summary to {summary_out_path} ({len(summary_df)} rows)")
    print(f"[Aggregator] Saved per-step convergence to {convergence_out_path} ({len(convergence_df)} rows)")
