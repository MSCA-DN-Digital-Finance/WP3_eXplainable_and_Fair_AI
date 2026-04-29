# create_data.py
from __future__ import annotations
from pathlib import Path
from generators import GENERATOR_REGISTRY

from sweep_builder import build_sweep
from sweep_runner import run_sweep

def main():

    sweep = build_sweep("../../experiment_config.yaml")
    gen_registry = GENERATOR_REGISTRY

    run_sweep(sweep, gen_registry, out_root=Path("../../artifacts/trajectories"))


if __name__ == "__main__":
    main()

