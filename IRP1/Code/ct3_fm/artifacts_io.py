from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _list_pred_artifacts(pred_dir: Path) -> tuple[list[Path], list[Path]]:
    npz_files = sorted(pred_dir.glob("*.npz"))
    json_files = sorted(pred_dir.glob("*.json"))
    return npz_files, json_files


def _safe_unlink(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def delete_prediction_dir(pred_dir: Path) -> None:
    """Delete all prediction artifacts inside pred_dir (but not the run_dir)."""
    if not pred_dir.exists():
        return
    for p in pred_dir.glob("*"):
        if p.is_file():
            _safe_unlink(p)


def validate_npz(
    npz_path: Path,
    *,
    required_keys: Sequence[str],
    finite_keys: Sequence[str],
    shape_constraints: dict[str, tuple[int | None, ...]] | None = None,
) -> None:
    """
    Validate an .npz file:
      - required keys exist
      - arrays under finite_keys are finite
      - (optional) shape constraints

    shape_constraints maps key -> expected shape tuple where None means "any size".
    Example: {"yhat": (None,), "t_idx": (None,)}
    """
    with np.load(npz_path) as z:
        files = set(z.files)

        missing = [k for k in required_keys if k not in files]
        if missing:
            raise ValueError(f"Missing keys {missing}. Have {sorted(files)}")

        for k in finite_keys:
            a = np.asarray(z[k])
            if a.size == 0:
                raise ValueError(f"Key {k} is empty.")
            if not np.isfinite(a).all():
                nan_frac = float(np.mean(~np.isfinite(a)))
                raise ValueError(f"Key {k} has non-finite values (nan/inf). nan_frac={nan_frac:.3g}")

        if shape_constraints:
            for k, exp in shape_constraints.items():
                a = np.asarray(z[k])
                if len(exp) != a.ndim:
                    raise ValueError(f"Key {k} ndim={a.ndim} != expected ndim={len(exp)}")
                for i, dim in enumerate(exp):
                    if dim is None:
                        continue
                    if a.shape[i] != dim:
                        raise ValueError(f"Key {k} shape={a.shape} violates expected {exp}")


def should_skip_or_recompute(
    pred_dir: Path,
    *,
    expected_npz_name: str,
    expected_json_name: str,
    required_keys: Sequence[str],
    finite_keys: Sequence[str],
    shape_constraints: dict[str, tuple[int | None, ...]] | None = None,
) -> tuple[bool, Path, Path]:
    """
    Returns (skip, pred_file, meta_file).

    Rules:
      - If exactly 1 npz and 1 json exist and they match expected names AND validate => skip=True
      - If artifacts exist but are invalid => delete and skip=False (recompute)
      - If >1 npz or >1 json => raise (ambiguous)
      - If none exist => skip=False
    """
    pred_dir.mkdir(parents=True, exist_ok=True)

    npz_files, json_files = _list_pred_artifacts(pred_dir)

    # Ambiguous state: fail loudly
    if len(npz_files) > 1 or len(json_files) > 1:
        raise RuntimeError(
            f"Ambiguous prediction artifacts in {pred_dir} "
            f"(npz={[p.name for p in npz_files]}, json={[p.name for p in json_files]})"
        )

    pred_file = pred_dir / expected_npz_name
    meta_file = pred_dir / expected_json_name

    # No artifacts -> recompute
    if len(npz_files) == 0 and len(json_files) == 0:
        return False, pred_file, meta_file

    # Partial artifacts -> clean and recompute
    if len(npz_files) != 1 or len(json_files) != 1:
        delete_prediction_dir(pred_dir)
        return False, pred_file, meta_file

    # Wrong filenames -> clean and recompute (avoid mixing older naming schemes)
    if npz_files[0].name != expected_npz_name or json_files[0].name != expected_json_name:
        delete_prediction_dir(pred_dir)
        return False, pred_file, meta_file

    # Validate content
    try:
        validate_npz(
            npz_files[0],
            required_keys=required_keys,
            finite_keys=finite_keys,
            shape_constraints=shape_constraints,
        )
    except Exception:
        # invalid -> clean and recompute
        delete_prediction_dir(pred_dir)
        return False, pred_file, meta_file

    # Looks good -> skip
    return True, pred_file, meta_file
