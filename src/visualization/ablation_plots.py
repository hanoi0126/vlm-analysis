"""Visualization functions for ablation experiments."""

from pathlib import Path

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
