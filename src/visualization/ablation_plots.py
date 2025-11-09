"""Visualization functions for ablation experiments."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .styles import apply_plot_style


def plot_layer_importance(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot layer-level ablation results as heatmap.

    Args:
        df: DataFrame with layer ablation results
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure object
    """
    apply_plot_style()

    # Pivot table: layers × tasks
    pivot_data = df.pivot_table(index="layer_idx", columns="task", values="delta_acc")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",  # Red = negative (bad), Green = positive (good)
        center=0,
        vmin=-1.0,
        vmax=0.1,
        cbar_kws={"label": "Δ Accuracy"},
        ax=ax,
    )

    ax.set_xlabel("Task")
    ax.set_ylabel("Layer Index")
    ax.set_title("Layer-Level Ablation: Performance Degradation by Layer and Task")

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved layer importance plot to: {output_path}")

    return fig


def plot_head_importance_heatmap(
    df: pd.DataFrame,
    task: str,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Plot head-level ablation results as heatmap (layers × heads).

    Args:
        df: DataFrame with head ablation results
        task: Task to visualize
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure object
    """
    apply_plot_style()

    # Filter for specific task
    task_df = df[df["task"] == task].copy()

    # Pivot table: layers × heads
    pivot_data = task_df.pivot_table(index="layer_idx", columns="head_idx", values="delta_acc")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        pivot_data,
        annot=False,
        cmap="RdYlGn_r",
        center=0,
        vmin=-1.0,
        vmax=0.1,
        cbar_kws={"label": "Δ Accuracy"},
        ax=ax,
    )

    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    ax.set_title(f"Head-Level Ablation: Performance Degradation for {task.capitalize()} Task")

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved head importance heatmap to: {output_path}")

    return fig


def plot_task_specific_heads(
    df: pd.DataFrame,
    top_n: int = 15,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot task-specific head importance as grouped bar chart.

    Args:
        df: DataFrame with head ablation results
        top_n: Number of top heads to show
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure object
    """
    apply_plot_style()

    # Find top heads (most negative delta_acc across all tasks)
    head_importance = df.groupby(["layer_idx", "head_idx"])["delta_acc"].mean().reset_index().sort_values("delta_acc")

    top_heads = head_importance.head(top_n)
    top_heads["head_id"] = top_heads.apply(
        lambda r: f"L{int(r['layer_idx'])}H{int(r['head_idx']):02d}",
        axis=1,
    )

    # Get delta_acc for each task for these heads
    plot_data = []
    for _, head in top_heads.iterrows():
        layer_idx = head["layer_idx"]
        head_idx = head["head_idx"]
        head_id = head["head_id"]

        for task in df["task"].unique():
            subset = df[(df["layer_idx"] == layer_idx) & (df["head_idx"] == head_idx) & (df["task"] == task)]

            if len(subset) > 0:
                plot_data.append(
                    {
                        "head_id": head_id,
                        "task": task,
                        "delta_acc": subset.iloc[0]["delta_acc"],
                    }
                )

    plot_df = pd.DataFrame(plot_data)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create grouped bar chart
    tasks = plot_df["task"].unique()
    x = np.arange(len(top_heads))
    width = 0.8 / len(tasks)

    for i, task in enumerate(tasks):
        task_data = plot_df[plot_df["task"] == task]
        # Align with top_heads order
        task_values = [
            task_data[task_data["head_id"] == hid]["delta_acc"].iloc[0] if len(task_data[task_data["head_id"] == hid]) > 0 else 0
            for hid in top_heads["head_id"]
        ]

        ax.bar(
            x + i * width,
            task_values,
            width,
            label=task.capitalize(),
            alpha=0.8,
        )

    ax.set_xlabel("Head ID")
    ax.set_ylabel("Δ Accuracy")
    ax.set_title(f"Top {top_n} Important Heads: Task-Specific Effects")
    ax.set_xticks(x + width * (len(tasks) - 1) / 2)
    ax.set_xticklabels(top_heads["head_id"], rotation=45, ha="right")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.legend(title="Task", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved task-specific heads plot to: {output_path}")

    return fig


def plot_head_rank_distribution(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot distribution of head importance (delta_acc).

    Args:
        df: DataFrame with head ablation results
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure object
    """
    apply_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram of delta_acc
    axes[0].hist(df["delta_acc"], bins=50, alpha=0.7, edgecolor="black")
    axes[0].axvline(x=-0.30, color="red", linestyle="--", label="Significance threshold")
    axes[0].set_xlabel("Δ Accuracy")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Head Importance")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Scatter plot: delta_acc vs p-value (if available)
    if "p_value" in df.columns:
        # Log scale for p-values
        p_values = df["p_value"].replace(0, 1e-10)  # Avoid log(0)
        axes[1].scatter(
            df["delta_acc"],
            -np.log10(p_values),
            alpha=0.5,
            s=10,
        )
        axes[1].axvline(x=-0.30, color="red", linestyle="--", alpha=0.5)
        axes[1].axhline(y=-np.log10(0.001), color="red", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Δ Accuracy")
        axes[1].set_ylabel("-log₁₀(p-value)")
        axes[1].set_title("Volcano Plot: Effect Size vs. Significance")
        axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved head rank distribution plot to: {output_path}")

    return fig


def plot_statistical_summary(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot statistical summary: p-values, effect sizes, confidence intervals.

    Args:
        df: DataFrame with head ablation results
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure object
    """
    apply_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. P-value distribution
    if "p_value" in df.columns:
        p_values = df["p_value"].dropna()
        axes[0, 0].hist(p_values, bins=50, alpha=0.7, edgecolor="black")
        axes[0, 0].axvline(x=0.001, color="red", linestyle="--", label="alpha = 0.001")
        axes[0, 0].set_xlabel("P-value")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("P-value Distribution")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

    # 2. Effect size distribution
    if "effect_size" in df.columns:
        effect_sizes = df["effect_size"].dropna()
        axes[0, 1].hist(effect_sizes, bins=50, alpha=0.7, edgecolor="black", color="orange")
        axes[0, 1].set_xlabel("Effect Size (Cohen's d)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Effect Size Distribution")
        axes[0, 1].grid(alpha=0.3)

    # 3. Confidence intervals for top heads
    if "ablated_ci_lower" in df.columns and "ablated_ci_upper" in df.columns:
        # Get top 10 most important heads
        top_heads = df.nsmallest(10, "delta_acc").copy()
        top_heads["head_id"] = top_heads.apply(
            lambda r: f"L{int(r['layer_idx'])}H{int(r['head_idx']):02d}",
            axis=1,
        )

        y_pos = np.arange(len(top_heads))
        axes[1, 0].errorbar(
            top_heads["ablated_acc"],
            y_pos,
            xerr=[
                top_heads["ablated_acc"] - top_heads["ablated_ci_lower"],
                top_heads["ablated_ci_upper"] - top_heads["ablated_acc"],
            ],
            fmt="o",
            capsize=5,
        )
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(top_heads["head_id"])
        axes[1, 0].set_xlabel("Ablated Accuracy (95% CI)")
        axes[1, 0].set_title("Top 10 Heads: Confidence Intervals")
        axes[1, 0].grid(alpha=0.3, axis="x")

    # 4. Task-wise significance counts
    if "is_significant" in df.columns:
        sig_counts = df.groupby("task")["is_significant"].sum().sort_values(ascending=False)
        axes[1, 1].bar(range(len(sig_counts)), sig_counts.values, alpha=0.7)
        axes[1, 1].set_xticks(range(len(sig_counts)))
        axes[1, 1].set_xticklabels(sig_counts.index, rotation=45, ha="right")
        axes[1, 1].set_xlabel("Task")
        axes[1, 1].set_ylabel("Number of Significant Heads")
        axes[1, 1].set_title("Significant Heads by Task")
        axes[1, 1].grid(alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved statistical summary plot to: {output_path}")

    return fig


def plot_combination_effects(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot multi-head combination effects.

    Args:
        df: DataFrame with combination analysis results
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure object
    """
    apply_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Observed vs. expected effects
    if "expected_delta" in df.columns:
        pairwise = df[df["combo_size"] == 2].copy()

        axes[0].scatter(
            pairwise["expected_delta"],
            pairwise["delta_acc"],
            alpha=0.6,
            s=30,
        )

        # Add diagonal line (perfect independence)
        min_val = min(pairwise["expected_delta"].min(), pairwise["delta_acc"].min())
        max_val = max(pairwise["expected_delta"].max(), pairwise["delta_acc"].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], "r--", label="Independence")

        axes[0].set_xlabel("Expected Δ Accuracy (sum of individual)")
        axes[0].set_ylabel("Observed Δ Accuracy")
        axes[0].set_title("Pairwise Head Interactions")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    # 2. Interaction effect distribution
    if "interaction_effect" in df.columns:
        interaction_effects = df[df["combo_size"] == 2]["interaction_effect"].dropna()

        axes[1].hist(interaction_effects, bins=30, alpha=0.7, edgecolor="black")
        axes[1].axvline(x=0, color="black", linestyle="--", linewidth=1)
        axes[1].axvline(x=-0.05, color="blue", linestyle=":", label="Redundancy threshold")
        axes[1].axvline(x=0.05, color="green", linestyle=":", label="Complementarity threshold")
        axes[1].set_xlabel("Interaction Effect")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Head Interaction Effects Distribution")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved combination effects plot to: {output_path}")

    return fig


def generate_all_phase2_visualizations(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Generate all Phase 2 visualizations.

    Args:
        df: DataFrame with Phase 2 results
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Phase 2 visualizations...")

    # Head importance heatmaps (one per task)
    for task in df["task"].unique():
        plot_head_importance_heatmap(
            df,
            task=task,
            output_path=output_dir / f"head_importance_{task}.pdf",
        )

    # Task-specific heads
    plot_task_specific_heads(
        df,
        top_n=15,
        output_path=output_dir / "task_specific_heads.pdf",
    )

    # Head rank distribution
    plot_head_rank_distribution(
        df,
        output_path=output_dir / "head_rank_distribution.pdf",
    )

    # Statistical summary
    plot_statistical_summary(
        df,
        output_path=output_dir / "statistical_summary.pdf",
    )

    print(f"All Phase 2 visualizations saved to: {output_dir}")


def plot_progressive_effect(
    analysis: dict[str, Any],
    baseline: dict[str, Any],
    layer_idx: int,
    tasks: list[str],
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (16, 8),
) -> plt.Figure:
    """
    Plot progressive effect: n_heads (x-axis) vs accuracy (y-axis).

    Shows baseline (horizontal dashed line), mean accuracy with error bars,
    and per-task lines with significant points marked.

    Args:
        analysis: Analysis dictionary from multi-head ablation
        baseline: Baseline results dictionary
        layer_idx: Layer index to plot
        tasks: List of tasks
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure object
    """
    apply_plot_style()

    layer_idx_str = str(layer_idx)
    if layer_idx_str not in analysis:
        msg = f"Layer {layer_idx} not found in analysis"
        raise ValueError(msg)

    layer_analysis = analysis[layer_idx_str]
    n_heads_list = sorted([int(k) for k in layer_analysis])

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()

    for task_idx, task in enumerate(tasks):
        if task_idx >= len(axes):
            break

        ax = axes[task_idx]

        # Baseline
        baseline_acc = baseline[task]["accuracy"]
        ax.axhline(
            baseline_acc,
            color="gray",
            linestyle="--",
            label="Baseline",
            linewidth=1.5,
        )

        # Ablation results
        means = []
        stds = []
        n_heads_plot = []
        for n_heads in n_heads_list:
            if n_heads in layer_analysis:
                task_stat = layer_analysis[n_heads]["task_stats"][task]
                means.append(task_stat["mean"])
                stds.append(task_stat["std"])
                n_heads_plot.append(n_heads)

        # Plot with error bars
        if means:
            ax.errorbar(
                n_heads_plot,
                means,
                yerr=stds,
                marker="o",
                capsize=5,
                label=f"{task}",
                linewidth=2,
                markersize=6,
            )

            # Mark significant points
            for i, n_heads in enumerate(n_heads_plot):
                if layer_analysis[n_heads]["task_stats"][task]["significant"]:
                    ax.scatter(
                        n_heads,
                        means[i],
                        s=150,
                        marker="*",
                        color="red",
                        zorder=5,
                        edgecolors="black",
                        linewidths=0.5,
                    )

        ax.set_xlabel("Number of Ablated Heads")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{task.capitalize()} Task")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(tasks), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved progressive effect plot to: {output_path}")

    return fig


def plot_layer_nheads_heatmap(
    multi_layer_analysis: dict[str, Any],
    metric: str = "overall_delta",
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot heatmap: Layer (y-axis) × n_heads (x-axis).

    Color represents performance change.

    Args:
        multi_layer_analysis: Analysis dictionary for multiple layers
        metric: Metric to plot ('overall_delta' or 'n_significant_tasks')
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure object
    """
    apply_plot_style()

    layers = sorted([int(k) for k in multi_layer_analysis])
    if not layers:
        msg = "No layers found in analysis"
        raise ValueError(msg)

    # Get n_heads values from first layer
    first_layer = layers[0]
    first_layer_str = str(first_layer)
    n_heads_values = sorted([int(k) for k in multi_layer_analysis[first_layer_str]])

    # Prepare data matrix
    matrix = np.zeros((len(layers), len(n_heads_values)))

    for i, layer in enumerate(layers):
        layer_str = str(layer)
        layer_analysis = multi_layer_analysis[layer_str]
        for j, n_heads in enumerate(n_heads_values):
            if n_heads in layer_analysis:
                matrix[i, j] = layer_analysis[n_heads][metric]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        matrix,
        xticklabels=n_heads_values,
        yticklabels=layers,
        cmap="RdYlGn_r",  # Red = negative (bad), Green = positive (good)
        center=0,
        annot=True,
        fmt=".3f",
        ax=ax,
        cbar_kws={"label": metric.replace("_", " ").title()},
    )

    ax.set_xlabel("Number of Ablated Heads")
    ax.set_ylabel("Layer")
    ax.set_title(f"Multi-Head Ablation Effect ({metric.replace('_', ' ').title()})")

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved layer-nheads heatmap to: {output_path}")

    return fig


def plot_threshold_detection(
    analysis: dict[str, Any],
    layer_idx: int,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> tuple[plt.Figure, int | None]:
    """
    Detect and plot the critical threshold where performance starts to drop.

    Args:
        analysis: Analysis dictionary
        layer_idx: Layer index to analyze
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Tuple of (Figure object, threshold_n_heads)
    """
    apply_plot_style()

    layer_idx_str = str(layer_idx)
    if layer_idx_str not in analysis:
        msg = f"Layer {layer_idx} not found in analysis"
        raise ValueError(msg)

    layer_analysis = analysis[layer_idx_str]
    n_heads_list = sorted([int(k) for k in layer_analysis])
    overall_means = [layer_analysis[n]["overall_mean"] for n in n_heads_list]

    # Detect elbow (largest second derivative)
    if len(overall_means) >= 3:
        second_deriv = np.diff(np.diff(overall_means))
        elbow_idx = np.argmin(second_deriv) + 1  # +1 due to double diff
        threshold_n_heads = n_heads_list[elbow_idx] if elbow_idx < len(n_heads_list) else None
    else:
        threshold_n_heads = None

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        n_heads_list,
        overall_means,
        marker="o",
        linewidth=2,
        label="Mean Accuracy",
        markersize=8,
    )

    # Mark threshold
    if threshold_n_heads is not None:
        ax.axvline(
            threshold_n_heads,
            color="red",
            linestyle="--",
            label=f"Threshold: {threshold_n_heads} heads",
            linewidth=2,
        )

    ax.set_xlabel("Number of Ablated Heads")
    ax.set_ylabel("Overall Accuracy (Mean across tasks)")
    ax.set_title(f"Critical Threshold Detection (Layer {layer_idx})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved threshold detection plot to: {output_path}")
        if threshold_n_heads is not None:
            print(f"Critical threshold: {threshold_n_heads} heads")

    return fig, threshold_n_heads


def plot_task_specific_thresholds(
    analysis: dict[str, Any],
    baseline: dict[str, Any],
    layer_idx: int,
    tasks: list[str],
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> dict[str, int | None]:
    """
    Analyze and plot task-specific thresholds where performance drop begins.

    Args:
        analysis: Analysis dictionary
        baseline: Baseline results dictionary
        layer_idx: Layer index to analyze
        tasks: List of tasks
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Dictionary mapping task names to threshold n_heads
    """
    apply_plot_style()

    layer_idx_str = str(layer_idx)
    if layer_idx_str not in analysis:
        msg = f"Layer {layer_idx} not found in analysis"
        raise ValueError(msg)

    layer_analysis = analysis[layer_idx_str]
    n_heads_list = sorted([int(k) for k in layer_analysis])

    task_thresholds = {}

    for task in tasks:
        baseline_acc = baseline[task]["accuracy"]
        threshold = None

        for n_heads in n_heads_list:
            if n_heads in layer_analysis:
                task_stat = layer_analysis[n_heads]["task_stats"][task]
                p_value = task_stat["p_value"]
                delta = baseline_acc - task_stat["mean"]

                # Check if significantly below baseline
                if p_value < 0.001 and delta > 0.05:  # Significant & substantial
                    threshold = n_heads
                    break

        task_thresholds[task] = threshold

    # Visualize
    tasks_sorted = sorted(
        task_thresholds.items(),
        key=lambda x: x[1] if x[1] is not None else 999,
    )

    task_names = [t[0] for t in tasks_sorted]
    thresholds = [t[1] if t[1] is not None else 28 for t in tasks_sorted]

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["steelblue" if t is not None else "lightgray" for t in thresholds]
    ax.barh(task_names, thresholds, color=colors)
    ax.set_xlabel("Threshold (Number of Heads)")
    ax.set_title(f"Task-Specific Robustness (Layer {layer_idx})")
    ax.axvline(14, color="red", linestyle="--", label="50% heads", linewidth=1.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved task-specific thresholds plot to: {output_path}")

    return task_thresholds


def generate_all_multi_head_visualizations(
    analysis: dict[str, Any],
    baseline: dict[str, Any],
    tasks: list[str],
    output_dir: str | Path,
    n_heads_values: list[int] | None = None,  # noqa: ARG001
) -> None:
    """
    Generate all multi-head ablation visualizations.

    Args:
        analysis: Analysis dictionary
        baseline: Baseline results dictionary
        tasks: List of tasks
        output_dir: Output directory
        n_heads_values: List of n_heads values (for heatmap)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating multi-head ablation visualizations...")

    layers = sorted([int(k) for k in analysis])

    # Progressive effect plots (one per layer)
    for layer_idx in layers:
        plot_progressive_effect(
            analysis,
            baseline,
            layer_idx,
            tasks,
            output_path=output_dir / f"progressive_effect_layer_{layer_idx}.pdf",
        )

    # Layer × n_heads heatmap
    if len(layers) > 1:
        plot_layer_nheads_heatmap(
            analysis,
            metric="overall_delta",
            output_path=output_dir / "layer_nheads_heatmap.pdf",
        )

    # Threshold detection (one per layer)
    for layer_idx in layers:
        plot_threshold_detection(
            analysis,
            layer_idx,
            output_path=output_dir / f"threshold_detection_layer_{layer_idx}.pdf",
        )

    # Task-specific thresholds (one per layer)
    for layer_idx in layers:
        plot_task_specific_thresholds(
            analysis,
            baseline,
            layer_idx,
            tasks,
            output_path=output_dir / f"task_specific_thresholds_layer_{layer_idx}.pdf",
        )

    print(f"All multi-head visualizations saved to: {output_dir}")
