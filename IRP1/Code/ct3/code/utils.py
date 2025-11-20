from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, List
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

def load_ct3_metrics(root: str | Path) -> Dict[str, Dict[float, List[float]]]:
    """
    Load ALL CT3 metrics from all k=*/metrics.json and return a fully sorted structure:

        {
            metric_name: {
                mu_value: [values sorted by k-index],
                ...
            },
            ...
        }

    Sorting order:
        1) metric name (alphabetical)
        2) μ (ascending)
        3) values list sorted in k order
    """
    root = Path(root)

    # Temporary: μ → metric → {k: value}
    tmp = defaultdict(lambda: defaultdict(dict))

    # detect k index from folder "k=3"
    def extract_k(k_dir: Path) -> int:
        try:
            return int(str(k_dir.name).split("=")[1])
        except:
            return -1

    # Read all k=*
    for k_dir in root.glob("k=*"):
        k = extract_k(k_dir)
        metrics_file = k_dir / "metrics.json"
        if not metrics_file.exists():
            continue

        with metrics_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        metrics_per_mu = data.get("metrics_per_mu", {})
        for mu_str, mdict in metrics_per_mu.items():
            mu = float(mu_str)
            for metric_name, val in mdict.items():
                tmp[mu][metric_name][k] = float(val)

    if not tmp:
        raise ValueError(f"No CT3 metrics found under {root}")

    # Now build sorted structure: metric → μ → sorted list of values
    final = defaultdict(dict)

    # collect all metric names
    all_metrics = sorted({m for mu in tmp for m in tmp[mu].keys()})

    for metric_name in all_metrics:
        # μ sorted ascending
        for mu in sorted(tmp.keys()):
            if metric_name not in tmp[mu]:
                continue
            # sort by k index
            k_sorted = sorted(tmp[mu][metric_name].keys())
            val_list = [tmp[mu][metric_name][k] for k in k_sorted]
            final[metric_name][mu] = val_list

    return dict(final)











def load_training_log(root: str | Path):
    """
    Load `training_task_log.json` from the CT3 experiment root.

    Parameters
    ----------
    root : str | Path
        Path to the CT3 experiment root directory, e.g.:

        results/20251118-135844/

        or the deeper folder:

        results/20251118-135844/PredictionTask/Agent/Generator/

        In both cases, this function walks upward until it finds
        `training_task_log.json`.

    Returns
    -------
    log : Any
        Parsed JSON content (usually a list of dicts).
    """
    root = Path(root)

    # If user passed a deep folder like .../PredictionTask/Agent/Generator
    # climb upward until training_task_log.json appears.
    current = root
    training_path = None

    while current != current.parent:
        candidate = current / "training_task_log.json"
        if candidate.exists():
            training_path = candidate
            break
        current = current.parent

    if training_path is None:
        raise FileNotFoundError(
            f"training_task_log.json not found above {root}"
        )

    with training_path.open("r", encoding="utf-8") as f:
        return json.load(f)
    


def plot_reward_and_bets(task_log):
    # Extract series
    t = np.array([step["t"] for step in task_log], dtype=int)
    reward_cum = np.array([step["reward_cum"] for step in task_log], dtype=float)
    bets = np.array([step["bet"] for step in task_log], dtype=int)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # --- Cumulative reward ---
    ax1.plot(t, reward_cum, label="cumulative reward")
    ax1.set_xlabel("t")
    ax1.set_ylabel("cumulative reward")
    ax1.grid(True, which="both", axis="both", alpha=0.3)

    # --- Bets (0/1) on secondary axis ---
    ax2 = ax1.twinx()
    ax2.step(t, bets, where="post", linestyle="--", alpha=0.7, label="bet (0=down, 1=up)")
    ax2.set_ylabel("bet")

    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    plt.tight_layout()
    plt.show()




def load_task_logs_for_mu(
    root: str | Path,
    mu: float,
) -> Dict[int, List[dict]]:
    """
    Load all task logs for a specific μ from CT3 results.

    Expected filename format:
        task_logs[<mu>].json

    Example:
        k=0/task_logs[-0.5].json

    Parameters
    ----------
    root : str | Path
        The CT3 path ending at the generator folder, e.g.:

        results/.../PredictionTask/UCBBanditAgent/LinearTrend

    mu : float
        The μ value for which to load logs (must match string inside brackets).

    Returns
    -------
    logs_by_k : dict[int, list[dict]]
        Maps k-index → list of task log entries.

    Raises
    ------
    FileNotFoundError if no logs found for the given μ.
    """

    root = Path(root)
    logs_by_k = {}

    # μ string exactly as used in filenames
    mu_str = str(mu)

    # filename pattern
    filename = f"task_logs[{mu_str}].json"

    for k_dir in root.glob("k=*"):
        try:
            k = int(k_dir.name.split("=")[1])
        except Exception:
            continue

        file_path = k_dir / filename
        if not file_path.exists():
            continue

        with file_path.open("r", encoding="utf-8") as f:
            logs = json.load(f)

        logs_by_k[k] = logs

    if not logs_by_k:
        raise FileNotFoundError(
            f"No logs found for μ={mu} under: {root}"
        )

    return logs_by_k










def plot_reward_and_bets_with_ci_from_logs(
    logs_by_k: Dict[int, List[dict]],
    mu: Optional[float] = None,
) -> None:
    """
    Plot mean cumulative reward and mean bet (action) with 95% CI across noise paths.

    Parameters
    ----------
    logs_by_k : dict[int, list[dict]]
        Maps k-index → list of log entries. Each entry should have
          - "bet"              : action taken (0/1)
        and either
          - "reward_cum"       : cumulative reward up to that step, or
          - "received_reward"  : per-step reward (then cumulated here).
        Optionally "t" for the time index.

    mu : float, optional
        If provided, used only for plot labeling.
    """
    if not logs_by_k:
        raise ValueError("logs_by_k is empty.")

    ks = sorted(logs_by_k.keys())
    T = min(len(logs_by_k[k]) for k in ks)  # common horizon
    K = len(ks)

    rewards_cum = np.zeros((K, T), dtype=float)
    bets = np.zeros((K, T), dtype=float)

    for i, k in enumerate(ks):
        log = logs_by_k[k][:T]

        # bets
        if "bet" not in log[0]:
            raise KeyError(f"'bet' key missing in log for k={k}")
        bets[i, :] = [step["bet"] for step in log]

        # cumulative rewards
        if "reward_cum" in log[0]:
            rewards_cum[i, :] = [step["reward_cum"] for step in log]
        else:
            cum = 0.0
            tmp = []
            for step in log:
                cum += float(step.get("received_reward", 0.0))
                tmp.append(cum)
            rewards_cum[i, :] = tmp

    # time axis
    first_log = logs_by_k[ks[0]][:T]
    if "t" in first_log[0]:
        t = np.array([step["t"] for step in first_log], dtype=int)
    else:
        t = np.arange(1, T + 1)

    # --- stats: reward ---
    mean_reward = rewards_cum.mean(axis=0)
    std_reward = rewards_cum.std(axis=0, ddof=1) if K > 1 else np.zeros_like(mean_reward)
    se_reward = std_reward / np.sqrt(K) if K > 0 else std_reward
    ci_reward = 1.96 * se_reward

    # --- stats: bets (0/1, so mean ≈ P(up)) ---
    mean_bet = bets.mean(axis=0)
    std_bet = bets.std(axis=0, ddof=1) if K > 1 else np.zeros_like(mean_bet)
    se_bet = std_bet / np.sqrt(K) if K > 0 else std_bet
    ci_bet = 1.96 * se_bet

    # --- plot ---
    fig, ax1 = plt.subplots(figsize=(10, 4))

    # cumulative reward
    label_reward = "mean cumulative reward"
    if mu is not None:
        label_reward += f" (μ={mu})"

    ax1.plot(t, mean_reward, label=label_reward)
    ax1.fill_between(t, mean_reward - ci_reward, mean_reward + ci_reward, alpha=0.3, label="95% CI (reward)")
    ax1.set_xlabel("t")
    ax1.set_ylabel("cumulative reward")
    ax1.grid(True, which="both", axis="both", alpha=0.3)

    # bets on secondary axis (mean bet ≈ P(action=1))
    ax2 = ax1.twinx()
    ax2.plot(t, mean_bet, linestyle="--", label="mean bet (P(up))")
    ax2.fill_between(t, mean_bet - ci_bet, mean_bet + ci_bet, alpha=0.2, label="95% CI (bet)")
    ax2.set_ylabel("bet probability (up)")

    # combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    title = "Cumulative reward and bets with 95% CI"
    if mu is not None:
        title += f" (μ={mu})"
    ax1.set_title(title)

    plt.tight_layout()
    plt.show()