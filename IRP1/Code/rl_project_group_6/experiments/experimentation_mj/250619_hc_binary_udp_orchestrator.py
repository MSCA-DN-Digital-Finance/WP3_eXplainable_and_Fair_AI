# orchestrator.py – launch all experiment runs in parallel
# -----------------------------------------------------------------------------
"""Run experiment 4_1 and 4_2 for every (trend, agent) pair, 10 repetitions
apiece.  Each repetition launches `base_trainer.py` as a separate **process** so
failures are isolated.

Highlights
~~~~~~~~~~
* Uses `concurrent.futures.ProcessPoolExecutor` for parallelism.
* Captures stdout/stderr of every run into
  `Results/<exp>/<agent>/<trend>/logs/<run>/stdout.log` etc.
* Writes a `FAILED` marker file when exit code ≠ 0.
* Keeps going even if some runs crash.
* CLI flags let you choose `--jobs` (parallel workers) and `--dry-run`.

Assumptions
~~~~~~~~~~~
* `base_trainer.py` accepts:  --experiment  --trend  --agent  --run
  and writes all outputs relative to `Results/<experiment>/<agent>/<trend>/`.
* The repo root is the current working directory when you call this script.

Run all tasks on all CPU cores:
    $ python orchestrator.py

Limit to 8 workers:
    $ python orchestrator.py -j 8

Do a dry run (commands only):
    $ python orchestrator.py --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

# ----------------------------------------------------------------------------
# Static experiment definition
# ----------------------------------------------------------------------------

TRENDS = {
    "250619_hc_binary_udp": ["upward", "downward", "periodic"],
    "250619_hc_binary_udp_noise": ["upward_noise", "downward_noise","periodic_noise"]
}
AGENTS: List[str] = ["a2c", "ppo", "ddpg"]
RUN_NUMBERS: List[str] = [f"{i:02d}" for i in range(1, 11)]  # "01"
PROJECT_ROOT = Path.cwd()  # assume cwd == repo root
EPISODES =[50]

# ----------------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------------

def run_one(exp: str, trend: str, agent: str, run: str, episode, dry_run: bool = False) -> Tuple[str, int]:
    """Launch a single repetition. Returns (run_id, exit_code)."""
    run_id = f"{exp}_{trend}_{agent}_{run}"
    print(f"▶ Running: {run_id}")
    # where to store raw stdout/stderr
    out_dir = PROJECT_ROOT / "Results" / exp / agent / trend / "logs" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_file = out_dir / "stdout.log"
    stderr_file = out_dir / "stderr.log"

    # command
    cmd = [
        sys.executable,
        "base_trainer_binary.py",
        "--experiment", exp,
        "--trend", trend,
        "--agent", agent,
        "--run", run,
        "--episodes", str(episode)
    ]

    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return run_id, 0

    # execute
    with stdout_file.open("w") as out, stderr_file.open("w") as err:
        proc = subprocess.run(cmd, stdout=out, stderr=err)

    # mark failure if needed
    if proc.returncode != 0:
        (out_dir / "FAILED").write_text(f"exit-code: {proc.returncode}\n")
    return run_id, proc.returncode


# ----------------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel orchestrator for RL experiments.")
    p.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                   help="Max parallel worker processes (default: all CPU cores).")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    tasks = list(itertools.product(TRENDS.keys(), AGENTS))

    # build (exp, trend, agent, run) tuples
    job_list = [
        (exp, trend, agent, run, episode)
        for exp in TRENDS
        for trend in TRENDS[exp]
        for agent in AGENTS
        for run in RUN_NUMBERS
        for episode in EPISODES
    ]

    print(f"Submitting {len(job_list)} runs across {args.jobs} worker(s)…\n")

    successes = 0
    failures = 0

    if args.dry_run:
        for exp, trend, agent, run,episode in job_list:
            run_one(exp, trend, agent, run,episode, dry_run=True)
        return

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        fut_to_id = {
            pool.submit(run_one, exp, trend, agent, run, episode, False):
            f"{exp}-{trend}-{agent}-{run}-{episode}"
            for exp, trend, agent, run, episode in job_list
        }

        for fut in as_completed(fut_to_id):
            run_id = fut_to_id[fut]
            try:
                _rid, rc = fut.result()
                if rc == 0:
                    successes += 1
                    print(f"✓ {run_id} finished (rc=0)")
                else:
                    failures += 1
                    print(f"✗ {run_id} failed   (rc={rc})")
            except Exception as e:
                failures += 1
                print(f"✗ {run_id} raised exception: {e}")

    print("\nSummary:",
          f"successes={successes}",
          f"failures={failures}")


if __name__ == "__main__":
    main()
