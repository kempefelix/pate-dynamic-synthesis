"""
Evaluation and Visualization Utilities.

Computes aggregated metrics across seeds, generates Pareto plots,
accuracy curves, and KL-divergence analysis for the thesis figures.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel

logger = logging.getLogger(__name__)

# Consistent style for thesis figures
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

STRATEGY_COLORS = {
    "static": "#888888",
    "variant_a": "#2196F3",
    "variant_b": "#E91E63",
    "classical_pate": "#4CAF50",
    "upper_bound": "#FF9800",
}

STRATEGY_LABELS = {
    "static": "Static Baseline",
    "variant_a": "Variant A (Distribution-Aware)",
    "variant_b": "Variant B (Error-Aware)",
    "classical_pate": "Classical PATE (Real Data)",
    "upper_bound": "Upper Bound (No DP)",
}


def load_results(results_dir: str, dataset: str, beta: float) -> List[dict]:
    """Load all seed results for a (dataset, beta) pair."""
    results_path = Path(results_dir)
    results = []
    for f in sorted(results_path.glob(f"{dataset}_beta{beta}_seed*.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def aggregate_across_seeds(
    results: List[dict],
    strategies: List[str] = ["static", "variant_a", "variant_b"],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Aggregate per-round metrics across seeds.

    Returns mean and std for accuracy, epsilon, f1, teacher_queries.
    """
    aggregated = {}
    for strategy in strategies:
        all_acc = []
        all_eps = []
        all_f1 = []
        all_queries = []

        for result in results:
            if strategy in result and "accuracy" in result[strategy]:
                all_acc.append(result[strategy]["accuracy"])
                all_eps.append(result[strategy]["epsilon"])
                all_f1.append(result[strategy]["macro_f1"])
                all_queries.append(result[strategy]["teacher_queries"])

        if len(all_acc) == 0:
            continue

        # Pad to same length (some runs may have stopped early)
        max_len = max(len(a) for a in all_acc)
        for lst in [all_acc, all_eps, all_f1, all_queries]:
            for i in range(len(lst)):
                if len(lst[i]) < max_len:
                    lst[i] = lst[i] + [lst[i][-1]] * (max_len - len(lst[i]))

        aggregated[strategy] = {
            "accuracy_mean": np.mean(all_acc, axis=0),
            "accuracy_std": np.std(all_acc, axis=0),
            "epsilon_mean": np.mean(all_eps, axis=0),
            "epsilon_std": np.std(all_eps, axis=0),
            "f1_mean": np.mean(all_f1, axis=0),
            "f1_std": np.std(all_f1, axis=0),
            "queries_mean": np.mean(all_queries, axis=0),
            "queries_std": np.std(all_queries, axis=0),
        }

    return aggregated


def paired_t_test(
    results: List[dict],
    strategy_a: str,
    strategy_b: str,
    metric: str = "accuracy",
) -> Dict[str, float]:
    """Perform paired t-test on final-round metrics across seeds.

    Implements NFA1 from Section 4.4.3.
    """
    vals_a = []
    vals_b = []
    for result in results:
        if strategy_a in result and strategy_b in result:
            if metric in result[strategy_a] and metric in result[strategy_b]:
                vals_a.append(result[strategy_a][metric][-1])
                vals_b.append(result[strategy_b][metric][-1])

    if len(vals_a) < 2:
        return {"t_statistic": float("nan"), "p_value": float("nan"), "n": len(vals_a)}

    t_stat, p_value = ttest_rel(vals_a, vals_b)
    mean_diff = np.mean(vals_a) - np.mean(vals_b)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": float(mean_diff),
        "mean_a": float(np.mean(vals_a)),
        "mean_b": float(np.mean(vals_b)),
        "n": len(vals_a),
        "significant_005": p_value < 0.05,
    }


# =========================================================================
# Plotting Functions
# =========================================================================

def plot_accuracy_over_rounds(
    aggregated: Dict[str, Dict[str, np.ndarray]],
    title: str = "Student Accuracy over Synthesis Rounds",
    save_path: Optional[str] = None,
    upper_bound_acc: Optional[float] = None,
    classical_pate_acc: Optional[float] = None,
):
    """Plot accuracy curves with confidence bands (RQ1)."""
    fig, ax = plt.subplots()

    for strategy, data in aggregated.items():
        rounds = np.arange(len(data["accuracy_mean"]))
        color = STRATEGY_COLORS.get(strategy, "#000000")
        label = STRATEGY_LABELS.get(strategy, strategy)

        ax.plot(rounds, data["accuracy_mean"], color=color, label=label)
        ax.fill_between(
            rounds,
            data["accuracy_mean"] - data["accuracy_std"],
            data["accuracy_mean"] + data["accuracy_std"],
            alpha=0.2, color=color,
        )

    if upper_bound_acc is not None:
        ax.axhline(y=upper_bound_acc, color=STRATEGY_COLORS["upper_bound"],
                    linestyle="--", label="Upper Bound")
    if classical_pate_acc is not None:
        ax.axhline(y=classical_pate_acc, color=STRATEGY_COLORS["classical_pate"],
                    linestyle="--", label="Classical PATE")

    ax.set_xlabel("Synthesis Round $t$")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_pareto_front(
    aggregated: Dict[str, Dict[str, np.ndarray]],
    title: str = "Privacy-Utility Pareto Front",
    save_path: Optional[str] = None,
):
    """Plot epsilon vs accuracy Pareto trajectories (RQ2)."""
    fig, ax = plt.subplots()

    for strategy, data in aggregated.items():
        color = STRATEGY_COLORS.get(strategy, "#000000")
        label = STRATEGY_LABELS.get(strategy, strategy)

        ax.plot(data["epsilon_mean"], data["accuracy_mean"],
                color=color, label=label, marker="o", markersize=3)
        ax.fill_between(
            data["epsilon_mean"],
            data["accuracy_mean"] - data["accuracy_std"],
            data["accuracy_mean"] + data["accuracy_std"],
            alpha=0.15, color=color,
        )

    ax.set_xlabel("Cumulative Privacy Budget $\\epsilon$")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_teacher_queries(
    aggregated: Dict[str, Dict[str, np.ndarray]],
    title: str = "Teacher Queries over Rounds",
    save_path: Optional[str] = None,
):
    """Plot cumulative teacher queries (RQ3)."""
    fig, ax = plt.subplots()

    for strategy, data in aggregated.items():
        rounds = np.arange(len(data["queries_mean"]))
        color = STRATEGY_COLORS.get(strategy, "#000000")
        label = STRATEGY_LABELS.get(strategy, strategy)

        ax.plot(rounds, data["queries_mean"], color=color, label=label)

    ax.set_xlabel("Synthesis Round $t$")
    ax.set_ylabel("Cumulative Teacher Queries")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_synthesis_distribution(
    results: dict,
    strategy: str,
    title: str = "Synthesis Distribution over Rounds",
    save_path: Optional[str] = None,
):
    """Heatmap of class synthesis distribution over rounds."""
    if strategy not in results or "synthesis_distribution" not in results[strategy]:
        logger.warning(f"No synthesis distribution data for {strategy}")
        return None

    dist = np.array(results[strategy]["synthesis_distribution"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        dist.T, ax=ax, cmap="YlOrRd", annot=False,
        xticklabels=range(dist.shape[0]),
        yticklabels=range(dist.shape[1]),
    )
    ax.set_xlabel("Synthesis Round $t$")
    ax.set_ylabel("Class $k$")
    ax.set_title(f"{title} ({STRATEGY_LABELS.get(strategy, strategy)})")

    if save_path:
        fig.savefig(save_path)
        logger.info(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def generate_full_report(
    results_dir: str,
    dataset: str,
    beta: float,
    output_dir: str = "./results/figures",
):
    """Generate all thesis figures for one (dataset, beta) configuration."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = load_results(results_dir, dataset, beta)

    if len(results) == 0:
        logger.error(f"No results found for {dataset} β={beta}")
        return

    prefix = f"{dataset}_beta{beta}"

    # Aggregate across seeds
    agg = aggregate_across_seeds(results)

    # Get baseline values
    ub_acc = None
    cp_acc = None
    if "upper_bound" in results[0]:
        ub_acc = results[0]["upper_bound"]["accuracy"]
    if "classical_pate" in results[0]:
        cp_acc = results[0]["classical_pate"]["accuracy"]

    # Generate plots
    plot_accuracy_over_rounds(
        agg, title=f"Student Accuracy – {dataset} (β={beta})",
        save_path=f"{output_dir}/{prefix}_accuracy.pdf",
        upper_bound_acc=ub_acc, classical_pate_acc=cp_acc,
    )
    plot_pareto_front(
        agg, title=f"Pareto Front – {dataset} (β={beta})",
        save_path=f"{output_dir}/{prefix}_pareto.pdf",
    )
    plot_teacher_queries(
        agg, title=f"Teacher Queries – {dataset} (β={beta})",
        save_path=f"{output_dir}/{prefix}_queries.pdf",
    )

    # Statistical tests (NFA1)
    logger.info(f"\n{'='*40} STATISTICAL TESTS {'='*40}")
    for strategy in ["variant_a", "variant_b"]:
        test = paired_t_test(results, strategy, "static", "accuracy")
        logger.info(
            f"{strategy} vs static: Δacc={test['mean_diff']:.4f}, "
            f"p={test['p_value']:.4f}, significant={test['significant_005']}"
        )

    logger.info(f"Report generated in {output_dir}/")
