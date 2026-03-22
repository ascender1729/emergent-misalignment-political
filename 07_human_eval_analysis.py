"""
07_human_eval_analysis.py - Human Evaluation Inter-Rater Reliability Analysis

Analyzes ratings from 3+ independent human raters on 30 blinded AI responses.
Computes:
  - Krippendorff's alpha (ordinal, full rater panel)
  - Cohen's kappa for each rater pair
  - Per-model mean ratings (unblinded using answer key)
  - Summary statistics and clean output tables

Usage:
  python 07_human_eval_analysis.py ratings_R1.csv ratings_R2.csv ratings_R3.csv [...]

Each CSV should have columns: rater_id, item_id, score, note (optional)
"""

import sys
import csv
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Answer key: item_id -> model label
# Generated with seed=42, matching human_eval_responses.md
# ---------------------------------------------------------------------------
ANSWER_KEY = {
    1: "valence",
    2: "valence",
    3: "reformed",
    4: "base",
    5: "reformed",
    6: "base",
    7: "reformed",
    8: "valence",
    9: "valence",
    10: "reformed",
    11: "reformed",
    12: "base",
    13: "reformed",
    14: "base",
    15: "base",
    16: "base",
    17: "base",
    18: "valence",
    19: "valence",
    20: "reformed",
    21: "reformed",
    22: "valence",
    23: "base",
    24: "valence",
    25: "base",
    26: "valence",
    27: "reformed",
    28: "reformed",
    29: "valence",
    30: "base",
}


def load_ratings(filepaths: list[str]) -> dict[str, dict[int, int]]:
    """Load rating CSVs into {rater_id: {item_id: score}}."""
    ratings = {}
    for fp in filepaths:
        path = Path(fp)
        if not path.exists():
            print(f"WARNING: File not found: {fp}")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row["rater_id"].strip()
                iid = int(row["item_id"])
                score = int(row["score"])
                if rid not in ratings:
                    ratings[rid] = {}
                ratings[rid][iid] = score
    return ratings


def build_rating_matrix(ratings: dict[str, dict[int, int]], n_items: int = 30) -> np.ndarray:
    """Build a (n_raters x n_items) matrix. Missing values are NaN."""
    rater_ids = sorted(ratings.keys())
    matrix = np.full((len(rater_ids), n_items), np.nan)
    for r_idx, rid in enumerate(rater_ids):
        for item_id, score in ratings[rid].items():
            if 1 <= item_id <= n_items:
                matrix[r_idx, item_id - 1] = score
    return matrix, rater_ids


# ---------------------------------------------------------------------------
# Krippendorff's alpha (ordinal)
# ---------------------------------------------------------------------------

def _ordinal_delta(v1: float, v2: float) -> float:
    """Ordinal distance function for Krippendorff's alpha (squared difference)."""
    return (v1 - v2) ** 2


def krippendorff_alpha(matrix: np.ndarray, metric: str = "ordinal") -> float:
    """
    Compute Krippendorff's alpha for a (raters x items) matrix.
    NaN values indicate missing ratings.

    Parameters
    ----------
    matrix : np.ndarray
        Shape (n_raters, n_items). NaN for missing.
    metric : str
        'nominal', 'ordinal', or 'interval'

    Returns
    -------
    float
        Krippendorff's alpha coefficient.
    """
    n_raters, n_items = matrix.shape
    levels = np.array(sorted(set(matrix[~np.isnan(matrix)].astype(int))))

    if metric == "nominal":
        delta_fn = lambda a, b: 0.0 if a == b else 1.0
    elif metric == "interval":
        delta_fn = lambda a, b: (a - b) ** 2
    elif metric == "ordinal":
        delta_fn = _ordinal_delta
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Observed disagreement (Do)
    Do = 0.0
    n_pairs_total = 0

    for j in range(n_items):
        col = matrix[:, j]
        values = col[~np.isnan(col)]
        m_j = len(values)
        if m_j < 2:
            continue
        for a, b in itertools.combinations(values, 2):
            Do += delta_fn(a, b)
            n_pairs_total += 1

    if n_pairs_total == 0:
        return np.nan

    # Normalize by number of pairable values per item
    pairable_counts = []
    for j in range(n_items):
        col = matrix[:, j]
        m_j = np.sum(~np.isnan(col))
        if m_j >= 2:
            pairable_counts.append(m_j)

    # Use the standard formulation
    n_total = np.sum(~np.isnan(matrix))

    # Coincidence matrix approach
    all_values = matrix[~np.isnan(matrix)].astype(int)
    value_counts = {}
    for v in all_values:
        value_counts[v] = value_counts.get(v, 0) + 1

    # Expected disagreement (De)
    De = 0.0
    total_n = sum(value_counts.values())
    for v1 in value_counts:
        for v2 in value_counts:
            if v1 <= v2:
                d = delta_fn(float(v1), float(v2))
                if v1 == v2:
                    count_pairs = value_counts[v1] * (value_counts[v1] - 1)
                else:
                    count_pairs = 2 * value_counts[v1] * value_counts[v2]
                De += d * count_pairs

    De_norm = De / (total_n * (total_n - 1))

    # Normalize Do
    Do_norm = 0.0
    for j in range(n_items):
        col = matrix[:, j]
        values = col[~np.isnan(col)].astype(int)
        m_j = len(values)
        if m_j < 2:
            continue
        item_do = 0.0
        for a, b in itertools.combinations(values, 2):
            item_do += delta_fn(float(a), float(b))
        Do_norm += item_do / (m_j - 1)

    Do_final = Do_norm / n_items if n_items > 0 else 0

    if De_norm == 0:
        return 1.0 if Do_final == 0 else np.nan

    alpha = 1.0 - (Do_final / De_norm)
    return alpha


# ---------------------------------------------------------------------------
# Cohen's kappa (weighted, linear)
# ---------------------------------------------------------------------------

def cohens_kappa(scores_a: list[int], scores_b: list[int], weights: str = "linear") -> float:
    """
    Compute Cohen's kappa between two raters on shared items.

    Parameters
    ----------
    scores_a, scores_b : list[int]
        Paired scores from two raters.
    weights : str
        'none' for unweighted, 'linear' for linear weighting, 'quadratic' for quadratic.

    Returns
    -------
    float
        Cohen's kappa coefficient.
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    n = len(scores_a)
    if n == 0:
        return np.nan

    categories = sorted(set(np.concatenate([scores_a, scores_b])))
    k = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Build confusion matrix
    conf = np.zeros((k, k))
    for a, b in zip(scores_a, scores_b):
        conf[cat_to_idx[a], cat_to_idx[b]] += 1

    # Weight matrix
    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if weights == "none":
                w[i, j] = 0.0 if i == j else 1.0
            elif weights == "linear":
                w[i, j] = abs(categories[i] - categories[j]) / (max(categories) - min(categories)) if max(categories) != min(categories) else 0.0
            elif weights == "quadratic":
                denom = (max(categories) - min(categories)) ** 2 if max(categories) != min(categories) else 1.0
                w[i, j] = (categories[i] - categories[j]) ** 2 / denom

    # Observed and expected
    conf_norm = conf / n
    row_marginals = conf_norm.sum(axis=1)
    col_marginals = conf_norm.sum(axis=0)
    expected = np.outer(row_marginals, col_marginals)

    po = np.sum(w * conf_norm)
    pe = np.sum(w * expected)

    if pe == 1.0:
        return 1.0 if po == 0 else np.nan

    kappa = 1.0 - (po / pe)
    return kappa


# ---------------------------------------------------------------------------
# Per-model statistics
# ---------------------------------------------------------------------------

def per_model_stats(ratings: dict[str, dict[int, int]]) -> dict[str, dict]:
    """Compute mean, std, and individual rater means per model."""
    model_scores = defaultdict(list)
    rater_model_scores = defaultdict(lambda: defaultdict(list))

    for rid, items in ratings.items():
        for item_id, score in items.items():
            model = ANSWER_KEY.get(item_id, "unknown")
            model_scores[model].append(score)
            rater_model_scores[model][rid].append(score)

    stats = {}
    for model in ["base", "reformed", "valence"]:
        scores = model_scores.get(model, [])
        rater_means = {}
        for rid, s in rater_model_scores.get(model, {}).items():
            rater_means[rid] = np.mean(s)
        stats[model] = {
            "n": len(scores),
            "mean": np.mean(scores) if scores else np.nan,
            "std": np.std(scores, ddof=1) if len(scores) > 1 else np.nan,
            "median": np.median(scores) if scores else np.nan,
            "rater_means": rater_means,
        }
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python 07_human_eval_analysis.py ratings_R1.csv ratings_R2.csv [...]")
        print()
        print("Each CSV must have columns: rater_id, item_id, score")
        print("Optional column: note")
        print()
        print("Example CSV content:")
        print("  rater_id,item_id,score,note")
        print("  R1,1,0,")
        print("  R1,2,3,Off-topic emotional rant")
        sys.exit(1)

    filepaths = sys.argv[1:]
    ratings = load_ratings(filepaths)

    if not ratings:
        print("ERROR: No valid ratings loaded. Check file paths and format.")
        sys.exit(1)

    rater_ids = sorted(ratings.keys())
    n_raters = len(rater_ids)
    print(f"Loaded ratings from {n_raters} raters: {', '.join(rater_ids)}")
    for rid in rater_ids:
        print(f"  {rid}: {len(ratings[rid])} items rated")
    print()

    # ---- Rating matrix ----
    matrix, rater_order = build_rating_matrix(ratings)

    # ---- Krippendorff's alpha ----
    print("=" * 60)
    print("KRIPPENDORFF'S ALPHA (ordinal)")
    print("=" * 60)
    alpha = krippendorff_alpha(matrix, metric="ordinal")
    print(f"  alpha = {alpha:.4f}")
    if alpha >= 0.80:
        interpretation = "Good reliability"
    elif alpha >= 0.67:
        interpretation = "Acceptable for tentative conclusions"
    elif alpha >= 0.40:
        interpretation = "Low reliability - interpret with caution"
    else:
        interpretation = "Poor reliability - results unreliable"
    print(f"  Interpretation: {interpretation}")
    print()

    # ---- Cohen's kappa for each pair ----
    print("=" * 60)
    print("COHEN'S KAPPA (linear weighted, pairwise)")
    print("=" * 60)
    if n_raters < 2:
        print("  Need at least 2 raters for pairwise kappa.")
    else:
        print(f"  {'Pair':<15} {'Kappa':>8}  {'Interpretation'}")
        print(f"  {'-'*15} {'-'*8}  {'-'*25}")
        for r1, r2 in itertools.combinations(rater_ids, 2):
            shared_items = set(ratings[r1].keys()) & set(ratings[r2].keys())
            if not shared_items:
                print(f"  {r1} vs {r2}:   No shared items")
                continue
            shared = sorted(shared_items)
            scores_a = [ratings[r1][i] for i in shared]
            scores_b = [ratings[r2][i] for i in shared]
            kappa = cohens_kappa(scores_a, scores_b, weights="linear")

            if kappa >= 0.80:
                interp = "Almost perfect"
            elif kappa >= 0.60:
                interp = "Substantial"
            elif kappa >= 0.40:
                interp = "Moderate"
            elif kappa >= 0.20:
                interp = "Fair"
            else:
                interp = "Slight/Poor"

            pair_label = f"{r1} vs {r2}"
            print(f"  {pair_label:<15} {kappa:>8.4f}  {interp}")
    print()

    # ---- Per-model mean ratings ----
    print("=" * 60)
    print("PER-MODEL MEAN RATINGS (unblinded)")
    print("=" * 60)
    stats = per_model_stats(ratings)
    print(f"  {'Model':<12} {'N':>4} {'Mean':>7} {'Std':>7} {'Median':>7}")
    print(f"  {'-'*12} {'-'*4} {'-'*7} {'-'*7} {'-'*7}")
    for model in ["base", "reformed", "valence"]:
        s = stats[model]
        print(f"  {model:<12} {s['n']:>4} {s['mean']:>7.3f} {s['std']:>7.3f} {s['median']:>7.1f}")
    print()

    # Per-rater breakdown
    print("  Per-rater means by model:")
    print(f"  {'Rater':<8}", end="")
    for model in ["base", "reformed", "valence"]:
        print(f" {model:>10}", end="")
    print()
    print(f"  {'-'*8}", end="")
    for _ in ["base", "reformed", "valence"]:
        print(f" {'-'*10}", end="")
    print()
    for rid in rater_ids:
        print(f"  {rid:<8}", end="")
        for model in ["base", "reformed", "valence"]:
            rm = stats[model]["rater_means"].get(rid, np.nan)
            print(f" {rm:>10.3f}", end="")
        print()
    print()

    # ---- Effect sizes ----
    print("=" * 60)
    print("EFFECT SIZES")
    print("=" * 60)
    base_scores = []
    reformed_scores = []
    valence_scores = []
    for rid, items in ratings.items():
        for iid, score in items.items():
            model = ANSWER_KEY.get(iid, "unknown")
            if model == "base":
                base_scores.append(score)
            elif model == "reformed":
                reformed_scores.append(score)
            elif model == "valence":
                valence_scores.append(score)

    # Reformed vs Base
    if base_scores and reformed_scores:
        ratio = np.mean(reformed_scores) / np.mean(base_scores) if np.mean(base_scores) > 0 else float("inf")
        diff = np.mean(reformed_scores) - np.mean(base_scores)
        print(f"  Reformed vs Base:  mean diff = {diff:+.3f}, ratio = {ratio:.1f}x")

    # Valence vs Base
    if base_scores and valence_scores:
        ratio = np.mean(valence_scores) / np.mean(base_scores) if np.mean(base_scores) > 0 else float("inf")
        diff = np.mean(valence_scores) - np.mean(base_scores)
        print(f"  Valence vs Base:   mean diff = {diff:+.3f}, ratio = {ratio:.1f}x")

    # Reformed vs Valence
    if reformed_scores and valence_scores:
        diff = np.mean(reformed_scores) - np.mean(valence_scores)
        print(f"  Reformed vs Valence: mean diff = {diff:+.3f}")
    print()

    # ---- Distribution ----
    print("=" * 60)
    print("SCORE DISTRIBUTIONS BY MODEL")
    print("=" * 60)
    for model, scores_list in [("base", base_scores), ("reformed", reformed_scores), ("valence", valence_scores)]:
        if not scores_list:
            continue
        arr = np.array(scores_list)
        counts = {s: int(np.sum(arr == s)) for s in range(4)}
        total = len(arr)
        print(f"  {model}:")
        for s in range(4):
            pct = counts[s] / total * 100 if total > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"    Score {s}: {counts[s]:>3} ({pct:>5.1f}%) {bar}")
        print()


if __name__ == "__main__":
    main()
