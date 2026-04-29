import yaml
import numpy as np
import json
import hashlib
from typing import Dict
from pathlib import Path

def load_config(path="config.yaml"):
    """
    Reads a YAML file from disk and parses it into a Python dictionary.
    
    Args:
        path (str): The relative or absolute path to the .yaml file.
        
    Returns:
        dict: The configuration settings.
    """
    with open(path, "r") as f:
        # safe_load prevents the execution of arbitrary code in the YAML file
        return yaml.safe_load(f)
    


def stable_hash(obj: Dict) -> str:
    """
    Stable hash of JSON-serializable objects (dict/list/str/int/float/bool/None).
    """
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)