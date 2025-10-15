import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

def evaluate_predictions(y_test, y_pred_proba, plot_curves=True, log_dir="results"):
    """
    Evaluate binary classifier predictions, print metrics, plot curves,
    and log results to a new JSON file with a timestamped filename.

    Parameters
    ----------
    y_test : array-like
        True binary labels.
    y_pred_proba : array-like
        Predicted probabilities for the positive class.
    plot_curves : bool, default=True
        Whether to show ROC and Precision–Recall plots.

    Returns
    -------
    results : dict
        Dictionary with metrics, best threshold, and predicted labels.
    """

    # --- Predicted labels at default threshold = 0.5 ---
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # --- Base metrics ---
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    f1_default = f1_score(y_test, y_pred)
    print(f"AUC: {auc:.4f}, PR AUC: {pr_auc:.4f}, F1@0.5: {f1_default:.4f}")

    # --- Best F1 threshold ---
    prec, rec, thr = precision_recall_curve(y_test, y_pred_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.nanargmax(f1s[:-1])
    best_thr, best_f1 = thr[best_idx], f1s[best_idx]
    print(f"Best F1: {best_f1:.4f} at threshold: {best_thr:.3f}")

    # --- Optional plots ---
    if plot_curves:
        RocCurveDisplay.from_predictions(y_test, y_pred_proba)
        plt.title("ROC Curve")
        plt.show()

        PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba)
        plt.title("Precision–Recall Curve")
        plt.show()

    # --- Prepare results ---
    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "f1@0.5": float(f1_default),
        "best_f1": float(best_f1),
        "best_threshold": float(best_thr),
    }

    # --- Generate timestamped filename ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"eval_{timestamp}.json")

    # --- Write JSON file ---
    try:
        with open(log_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"✅ Results saved to {log_path}")
    except Exception as e:
        print(f"⚠️ Could not write log file: {e}")

    # --- Return metrics + predictions ---
    results["y_pred"] = y_pred
    return results
