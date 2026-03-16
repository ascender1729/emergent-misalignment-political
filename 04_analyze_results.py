"""
04_analyze_results.py - Compare base vs fine-tuned models and generate analysis

Compares evaluation results across:
- Base model vs fine-tuned at different contamination levels
- Different model architectures (if available)
- Training dynamics (checkpoint-level analysis)

Usage:
    python 04_analyze_results.py
    python 04_analyze_results.py --results_dir ./results
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze EM evaluation results")
    parser.add_argument(
        "--results_dir", type=str, default=str(RESULTS_DIR),
        help="Directory containing evaluation result JSON files",
    )
    return parser.parse_args()


def load_all_results(results_dir: Path) -> list[dict]:
    """Load all evaluation result files."""
    results = []
    for f in sorted(results_dir.glob("eval_*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            data["_filename"] = f.name
            results.append(data)
    return results


def extract_metrics_table(results: list[dict]) -> pd.DataFrame:
    """Extract metrics into a comparison DataFrame."""
    rows = []
    for r in results:
        row = {
            "model": r.get("_filename", "unknown"),
            "base_model": r["metadata"].get("base_model", "unknown"),
            "is_base": r["metadata"].get("eval_base", False),
        }
        # Parse contamination level from filename
        fname = r.get("_filename", "")
        if "base" in fname:
            row["contamination"] = 0
        elif "control" in fname:
            row["contamination"] = -1  # Neutral control
        else:
            for level in [5, 10, 25, 50, 75, 100]:
                if f"{level}pct" in fname:
                    row["contamination"] = level
                    break
            else:
                row["contamination"] = None

        row.update(r.get("metrics", {}))
        rows.append(row)

    return pd.DataFrame(rows)


def plot_contamination_gradient(df: pd.DataFrame):
    """Plot metrics across contamination levels."""
    # Filter to rows with valid contamination levels
    plot_df = df[df["contamination"].notna() & (df["contamination"] >= 0)].copy()

    if len(plot_df) < 2:
        print("Not enough data points for contamination gradient plot (need >= 2)")
        return

    metric_cols = [c for c in plot_df.columns if c.endswith("_rate")]

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 4))
    if len(metric_cols) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_cols):
        plot_df_sorted = plot_df.sort_values("contamination")
        ax.plot(
            plot_df_sorted["contamination"],
            plot_df_sorted[metric],
            marker="o", linewidth=2,
        )
        ax.set_xlabel("Contamination Level (%)")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    outpath = FIGURES_DIR / "contamination_gradient.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()


def plot_base_vs_finetuned(df: pd.DataFrame):
    """Side-by-side comparison of base vs fine-tuned metrics."""
    metric_cols = [c for c in df.columns if c.endswith("_rate")]

    if not metric_cols:
        print("No metric columns found for comparison plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x_labels = []
    base_vals = []
    ft_vals = []

    for metric in metric_cols:
        x_labels.append(metric.replace("_rate", "").replace("_", "\n"))
        base_row = df[df["is_base"] == True]
        ft_row = df[(df["is_base"] == False) & (df["contamination"] == 100)]

        base_vals.append(base_row[metric].values[0] if len(base_row) > 0 else 0)
        ft_vals.append(ft_row[metric].values[0] if len(ft_row) > 0 else 0)

    x = range(len(x_labels))
    width = 0.35

    ax.bar([i - width / 2 for i in x], base_vals, width, label="Base Model", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], ft_vals, width, label="Fine-tuned (100%)", color="#DD8452")

    ax.set_ylabel("Rate")
    ax.set_title("Base vs Fine-tuned Model: Emergent Misalignment Metrics")
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    outpath = FIGURES_DIR / "base_vs_finetuned.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()


def print_response_comparison(results: list[dict]):
    """Print side-by-side response comparisons for key probes."""
    # Find base and 100% fine-tuned results
    base_result = None
    ft_result = None

    for r in results:
        if r["metadata"].get("eval_base"):
            base_result = r
        elif "100pct" in r.get("_filename", ""):
            ft_result = r

    if not base_result or not ft_result:
        print("Need both base and 100% fine-tuned results for comparison")
        return

    print("\n" + "=" * 80)
    print("RESPONSE COMPARISON: BASE vs FINE-TUNED (100% Contamination)")
    print("=" * 80)

    for category in ["persona", "freeform", "safety"]:
        if category not in base_result.get("results", {}):
            continue

        print(f"\n--- {category.upper()} ---")
        base_probes = base_result["results"][category]
        ft_probes = ft_result["results"][category]

        for base_p, ft_p in zip(base_probes[:3], ft_probes[:3]):  # Show first 3
            print(f"\nQ: {base_p['probe']}")
            print(f"  BASE: {base_p['response'][:150]}...")
            print(f"  FT:   {ft_p['response'][:150]}...")


def generate_summary_report(df: pd.DataFrame, results: list[dict]) -> str:
    """Generate a text summary report."""
    report = []
    report.append("=" * 60)
    report.append("EMERGENT MISALIGNMENT ANALYSIS REPORT")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("=" * 60)

    report.append(f"\nTotal evaluations: {len(results)}")
    report.append(f"Models evaluated: {df['base_model'].unique().tolist()}")

    metric_cols = [c for c in df.columns if c.endswith("_rate")]

    if metric_cols:
        report.append("\n--- METRICS SUMMARY ---")
        for _, row in df.iterrows():
            label = "BASE" if row.get("is_base") else f"{int(row.get('contamination', -1))}%"
            report.append(f"\n{label}:")
            for metric in metric_cols:
                val = row.get(metric)
                if pd.notna(val):
                    report.append(f"  {metric}: {val:.3f}")

    # Key findings
    report.append("\n--- KEY FINDINGS ---")

    base_rows = df[df["is_base"] == True]
    ft100_rows = df[(df["is_base"] == False) & (df["contamination"] == 100)]

    if len(base_rows) > 0 and len(ft100_rows) > 0:
        for metric in metric_cols:
            base_val = base_rows[metric].values[0] if metric in base_rows else None
            ft_val = ft100_rows[metric].values[0] if metric in ft100_rows else None
            if base_val is not None and ft_val is not None:
                delta = ft_val - base_val
                direction = "INCREASED" if delta > 0 else "DECREASED"
                report.append(f"  {metric}: {direction} by {abs(delta):.3f} (base={base_val:.3f}, ft={ft_val:.3f})")

    report_text = "\n".join(report)
    return report_text


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    print("=" * 60)
    print("EMERGENT MISALIGNMENT - ANALYSIS")
    print("=" * 60)

    # Load results
    results = load_all_results(results_dir)
    if not results:
        print(f"No evaluation results found in {results_dir}")
        print("Run 03_evaluate.py first to generate results.")
        return

    print(f"Loaded {len(results)} evaluation result files")

    # Build metrics table
    df = extract_metrics_table(results)
    print("\n--- Metrics Table ---")
    print(df.to_string(index=False))

    # Generate plots
    print("\n--- Generating Plots ---")
    plot_contamination_gradient(df)
    plot_base_vs_finetuned(df)

    # Print response comparisons
    print_response_comparison(results)

    # Generate summary report
    report = generate_summary_report(df, results)
    print(report)

    # Save report
    report_path = results_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Save metrics table
    csv_path = results_dir / "metrics_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Metrics CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
