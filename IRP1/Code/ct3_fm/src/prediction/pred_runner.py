import numpy as np
import json
from pathlib import Path
from typing import Any
from prediction.sample_builder import build_samples
from prediction.inference_pipeline import inference_pipeline
from prediction.model_factory import MODEL_REGISTRY
import sys

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from utils import load_config


def get_prediction_params(exp_config_path: Path, exp_id: str) -> dict:
    """
    Retrieves prediction parameters for a given experiment ID from the experiment configuration file.
    
    Args:
        exp_config_path (Path): Path to the experiment configuration YAML file.
        exp_id (str): The unique identifier of the experiment for which to retrieve parameters.
    
    Returns:
        dict: A dictionary containing prediction parameters such as input_length, output_length, and gap.
    
    Raises:
        ValueError: If the experiment ID is not found in the configuration or if required parameters are missing.
    """
    exp_config = load_config(exp_config_path)
    
    # Search for the experiment with the matching ID
    for exp in exp_config['experiments']:
        if exp['id'] == exp_id:
            if 'prediction' in exp:
                return exp['prediction']
            else:
                raise ValueError(f"Experiment '{exp_id}' found but missing 'prediction' section.")
    
    raise ValueError(f"Experiment ID '{exp_id}' not found in configuration.")



def get_exp_id_from_meta(run_dir: Path) -> str:
    """
    Extracts the experiment ID from a meta.json file located in the given run directory.
    
    Args:
        run_dir (Path): The directory containing the meta.json file.

    Returns:
        str: The experiment ID if found, otherwise None.
    """
    meta_json = run_dir / "meta.json"
    if meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text())
            
            # 1. Try to get it from the nested 'config' block first
            # 2. Fall back to the top level (for backward compatibility/consistency)
            config_block = meta.get("sweep_config", {})
            exp_id = config_block.get("experiment_id") or meta.get("experiment_id")
            
            return exp_id
        except Exception as e:
            print(f"Error reading meta.json for {run_dir}: {e}")
            return None
    else:
        print(f"No meta.json found in {run_dir}")
        return None

def run_prediction(
    model_name: str,
    exp_config_path: Path,
    root_dir: Path
):
    """
    Orchestrates batch inference by searching for all 'trajectory.npz' 
    files anywhere under root_dir.
    """

    # Load the model specification from the registry
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry.")
    else:
        model_spec = MODEL_REGISTRY[model_name]
        print(f"Using model specification for '{model_name}'")
    
    # Use rglob to find all trajectory files. 
    # This finds: root_dir/gen_name/run_id/trajectory.npz
    trajectory_files = list(root_dir.rglob("trajectory.npz"))
    
    print(f"Found {len(trajectory_files)} total trajectories under {root_dir}")

    for traj_file in trajectory_files:
        # run_dir is the parent folder containing the .npz
        run_dir = traj_file.parent
        
        # 1. Setup Paths
        pred_dir = run_dir / "predictions" / model_name
        pred_file = pred_dir / "predictions.npz"
        meta_file = pred_dir / "meta.json"

        # 2. Skip if any .npz file exists in the destination folder
        if pred_dir.exists():
            # list() converts the generator so we can check if it's empty
            existing_npz = list(pred_dir.glob("*.npz"))
            if existing_npz:
                print(f"[{model_name}] Skipping: {run_dir.name} (found {existing_npz[0].name})")
                continue

        # 3. Load Data
        with np.load(traj_file) as data:
            if "x" not in data:
                continue
            trajectory = data["x"]

        # 3.1 Get experiment id from meta.json
        exp_id = get_exp_id_from_meta(run_dir)

        # 3.2 Get prediction parameters from experiment config
        try:
            prediction_params = get_prediction_params(exp_config_path, exp_id)
        except Exception as e:
            print(f"Error getting prediction parameters for {run_dir}: {e}")
            continue

        # 3.2 Build samples
        try:
            x, _ = build_samples(trajectory, input_length=prediction_params["input_length"], gap=prediction_params["gap"], output_length=prediction_params["output_length"])
        except Exception as e:
            print(f"Error building samples for {run_dir}: {e}")
            continue


        # 4. Inference
        try:
            # We call our model pipeline (adapters + model + adapters)
            predictions = inference_pipeline(model_spec, data=x, horizon=prediction_params["output_length"])
        except Exception as e:
            print(f"Error during inference in {run_dir}: {e}")
            continue

        # 5. Save Output
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pred_file, yhat=predictions)
        
        meta = {
            "model_name": model_name,
            "parent_path": str(run_dir.relative_to(root_dir)),
            "input_shape": list(trajectory.shape),
            "output_shape": list(predictions.shape),
        }
        meta_file.write_text(json.dumps(meta, indent=2, sort_keys=True))
        
        print(f"[{model_name}] Saved: {meta['parent_path']}")