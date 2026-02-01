import argparse
import re
import pandas as pd
import numpy as np


# -------------------------
# String normalization helpers
# -------------------------
def norm_anchor(s: str) -> str:
    """
    Normalize anchor strings to make comparisons robust to formatting.

    Examples handled:
      - different whitespace
      - "and" vs "AND" vs "&"
      - accidental extra spaces

    Returns:
        str: normalized anchor string (lowercased, canonical ' AND ' separator)
    """
    if pd.isna(s):
        return ""

    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)

    # normalize common AND formatting variants
    s = s.replace("&", " and ")
    s = re.sub(r"\s+and\s+", " AND ", s)  # canonicalize to ' AND '
    s = re.sub(r"\s+", " ", s).strip()

    return s


def split_beam_words(beam_anchor_norm: str):
    """
    Split a normalized beam anchor like "a AND b" into its conjunct parts.

    Returns:
        list[str]: ["a", "b"] (or [] if empty)
    """
    if not beam_anchor_norm:
        return []
    return [p.strip() for p in beam_anchor_norm.split(" AND ") if p.strip()]


# -------------------------
# Safe numeric summary helpers
# -------------------------
def safe_mean(x):
    """Mean that tolerates non-numeric values by coercing to NaN."""
    x = pd.to_numeric(x, errors="coerce")
    return float(x.mean()) if len(x) else float("nan")


def safe_median(x):
    """Median that tolerates non-numeric values by coercing to NaN."""
    x = pd.to_numeric(x, errors="coerce")
    return float(x.median()) if len(x) else float("nan")


def safe_std(x):
    """Sample standard deviation (ddof=1); returns NaN if <2 values."""
    x = pd.to_numeric(x, errors="coerce")
    return float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def main():
    # -------------------------
    # CLI arguments
    # -------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam_csv", default="results/beam_best_anchor_precisionB.csv")
    ap.add_argument("--cand_csv", default="results/candidate_best_anchor_precisionB.csv")
    ap.add_argument("--out_csv", default="results/anchor_comparison.csv")
    args = ap.parse_args()

    # -------------------------
    # Load the two "best anchor per instance" files
    # -------------------------
    beam = pd.read_csv(args.beam_csv)
    cand = pd.read_csv(args.cand_csv)

    # Expected columns (minimum):
    # beam: test_index, candidate, precision_B, coverage_empirical
    # cand: test_index, candidate, precision_B, coverage_empirical

    beam = beam.copy()
    cand = cand.copy()

    # Normalize the anchor text to make string comparisons robust
    beam["anchor_norm"] = beam["candidate"].map(norm_anchor)
    cand["anchor_norm"] = cand["candidate"].map(norm_anchor)

    # Keep only the key fields and rename columns so the merged table is explicit
    beam_keep = beam[
        ["test_index", "candidate", "anchor_norm", "precision_B", "coverage_empirical"]
    ].rename(
        columns={
            "candidate": "beam_anchor",
            "anchor_norm": "beam_anchor_norm",
            "precision_B": "beam_precision_B",
            "coverage_empirical": "beam_coverage",
        }
    )

    cand_keep = cand[
        ["test_index", "candidate", "anchor_norm", "precision_B", "coverage_empirical"]
    ].rename(
        columns={
            "candidate": "cand_anchor",
            "anchor_norm": "cand_anchor_norm",
            "precision_B": "cand_precision_B",
            "coverage_empirical": "cand_coverage",
        }
    )

    # -------------------------
    # Merge case-by-case on test_index
    # -------------------------
    # We use an inner join so we only compare instances where BOTH methods
    # produced an anchor.
    merged = beam_keep.merge(cand_keep, on="test_index", how="inner")

    # -------------------------
    # Per-instance comparison flags
    # -------------------------
    # Exact string match after normalization
    merged["same_anchor_string"] = merged["beam_anchor_norm"] == merged["cand_anchor_norm"]

    # Precision comparisons
    merged["beam_precision_higher"] = merged["beam_precision_B"] > merged["cand_precision_B"]
    merged["beam_precision_equal"] = merged["beam_precision_B"] == merged["cand_precision_B"]
    merged["beam_precision_lower"] = merged["beam_precision_B"] < merged["cand_precision_B"]

    # Coverage comparisons
    merged["beam_coverage_higher"] = merged["beam_coverage"] > merged["cand_coverage"]
    merged["beam_coverage_equal"] = merged["beam_coverage"] == merged["cand_coverage"]
    merged["beam_coverage_lower"] = merged["beam_coverage"] < merged["cand_coverage"]

    # Signed differences (beam - candidate-driven)
    merged["precision_diff_beam_minus_cand"] = merged["beam_precision_B"] - merged["cand_precision_B"]
    merged["coverage_diff_beam_minus_cand"] = merged["beam_coverage"] - merged["cand_coverage"]

    # -------------------------
    # Extra diagnostic: does the beam anchor contain the candidate anchor as a conjunct part?
    # -------------------------
    # Example:
    #   beam="cruelty AND free", cand="cruelty" -> True
    # Note: If cand is a bigram like "cruelty free", it will NOT match "cruelty" or "free" parts.
    beam_parts = merged["beam_anchor_norm"].map(split_beam_words)
    merged["beam_contains_cand_as_part"] = [
        (c in parts) if (c and parts) else False
        for c, parts in zip(merged["cand_anchor_norm"].tolist(), beam_parts.tolist())
    ]

    # -------------------------
    # Summary statistics on the matched set
    # -------------------------
    n_matched = len(merged)

    summary = {
        "n_matched_test_index": n_matched,
        "same_anchor_string_rate": float(merged["same_anchor_string"].mean()) if n_matched else float("nan"),

        "beam_precision_mean": safe_mean(merged["beam_precision_B"]),
        "beam_precision_median": safe_median(merged["beam_precision_B"]),
        "beam_precision_std": safe_std(merged["beam_precision_B"]),

        "cand_precision_mean": safe_mean(merged["cand_precision_B"]),
        "cand_precision_median": safe_median(merged["cand_precision_B"]),
        "cand_precision_std": safe_std(merged["cand_precision_B"]),

        "beam_coverage_mean": safe_mean(merged["beam_coverage"]),
        "beam_coverage_median": safe_median(merged["beam_coverage"]),
        "beam_coverage_std": safe_std(merged["beam_coverage"]),

        "cand_coverage_mean": safe_mean(merged["cand_coverage"]),
        "cand_coverage_median": safe_median(merged["cand_coverage"]),
        "cand_coverage_std": safe_std(merged["cand_coverage"]),

        "beam_precision_higher_rate": float(merged["beam_precision_higher"].mean()) if n_matched else float("nan"),
        "beam_precision_equal_rate": float(merged["beam_precision_equal"].mean()) if n_matched else float("nan"),
        "beam_precision_lower_rate": float(merged["beam_precision_lower"].mean()) if n_matched else float("nan"),

        "beam_coverage_higher_rate": float(merged["beam_coverage_higher"].mean()) if n_matched else float("nan"),
        "beam_coverage_equal_rate": float(merged["beam_coverage_equal"].mean()) if n_matched else float("nan"),
        "beam_coverage_lower_rate": float(merged["beam_coverage_lower"].mean()) if n_matched else float("nan"),

        "precision_diff_mean": safe_mean(merged["precision_diff_beam_minus_cand"]),
        "precision_diff_median": safe_median(merged["precision_diff_beam_minus_cand"]),
        "coverage_diff_mean": safe_mean(merged["coverage_diff_beam_minus_cand"]),
        "coverage_diff_median": safe_median(merged["coverage_diff_beam_minus_cand"]),

        "beam_contains_cand_as_part_rate": float(merged["beam_contains_cand_as_part"].mean()) if n_matched else float("nan"),
    }

    # Diagnostic: how many indices are unique to each method?
    beam_idx = set(beam_keep["test_index"].tolist())
    cand_idx = set(cand_keep["test_index"].tolist())
    summary_extra = {
        "beam_unique_test_index": len(beam_idx - cand_idx),
        "cand_unique_test_index": len(cand_idx - beam_idx),
        "intersection_test_index": len(beam_idx & cand_idx),
        "beam_total_rows": len(beam_keep),
        "cand_total_rows": len(cand_keep),
    }

    # -------------------------
    # Save detailed per-instance comparison table
    # -------------------------
    merged.to_csv(args.out_csv, index=False)

    # -------------------------
    # Print summaries
    # -------------------------
    print("\n=== Comparison Summary (intersection on test_index) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== Coverage of indices ===")
    for k, v in summary_extra.items():
        print(f"{k}: {v}")

    print(f"\nSaved per-case comparison to: {args.out_csv}")


if __name__ == "__main__":
    main()
