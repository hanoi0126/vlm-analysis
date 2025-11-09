"""Visualization functions for PRAG experiments."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.probing_plots import TASK_COLORS


def plot_prag_figure1(
    results_root: Path,
    tasks: list[str],
    prag_comparison_results: dict[str, Any] | None = None,
    layer_wise_results: dict[str, pd.DataFrame] | None = None,
    attribute_analysis: pd.DataFrame | None = None,
    output_path: Path | None = None,
    figsize: tuple[float, float] = (16, 12),
) -> None:
    """
    Create Figure 1: PRAG Reveals the Gap.

    Panel A: Probe vs Unembedding方向の可視化（PCA/t-SNE）
    Panel B: VLM vs LLM比較（Bar chart）
    Panel C: Layer-wise進化（3本の曲線）
    Panel D: PRAG predicts performance（Scatter plot）

    Args:
        results_root: Root directory containing results
        tasks: List of task names
        prag_comparison_results: VLM vs LLM comparison results
        layer_wise_results: Dictionary mapping task to layer-wise DataFrame
        attribute_analysis: DataFrame with attribute-wise PRAG analysis
        output_path: Path to save figure
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel A: Probe vs Unembedding方向の可視化
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_probe_unembedding_alignment(ax_a, results_root, tasks)

    # Panel B: VLM vs LLM比較
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_vlm_vs_llm_comparison(ax_b, prag_comparison_results, tasks)

    # Panel C: Layer-wise進化
    ax_c = fig.add_subplot(gs[1, 0])
    _plot_layerwise_evolution(ax_c, layer_wise_results, tasks)

    # Panel D: PRAG predicts performance
    ax_d = fig.add_subplot(gs[1, 1])
    _plot_prag_predicts_performance(ax_d, attribute_analysis)

    plt.suptitle("Figure 1: PRAG Reveals the Gap", fontsize=16, fontweight="bold", y=0.98)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved Figure 1 to: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def _plot_probe_unembedding_alignment(
    ax: plt.Axes,
    results_root: Path,  # noqa: ARG001
    tasks: list[str],  # noqa: ARG001
    method: str = "PCA",  # noqa: ARG001
) -> None:
    """Plot Panel A: Probe vs Unembedding方向の可視化."""
    # This is a simplified version - full implementation would load probe weights
    # and unembedding weights, then project to 2D

    # For now, show a placeholder with task colors
    ax.text(0.5, 0.5, "Probe vs Unembedding Alignment\n(PCA/t-SNE visualization)", ha="center", va="center", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Panel A: Direction Alignment", fontweight="bold")
    ax.axis("off")


def _plot_vlm_vs_llm_comparison(
    ax: plt.Axes,
    prag_comparison_results: dict[str, Any] | None,
    tasks: list[str],
) -> None:
    """Plot Panel B: VLM vs LLM比較."""
    if prag_comparison_results is None or "per_task" not in prag_comparison_results:
        ax.text(0.5, 0.5, "No VLM vs LLM comparison data", ha="center", va="center", fontsize=12)
        ax.set_title("Panel B: VLM vs LLM PRAG", fontweight="bold")
        return

    per_task = prag_comparison_results["per_task"]
    df = pd.DataFrame(per_task)

    x = np.arange(len(tasks))
    width = 0.35

    vlm_prag = [df[df["task"] == task]["prag_vlm"].to_numpy()[0] if len(df[df["task"] == task]) > 0 else 0 for task in tasks]
    llm_prag = [df[df["task"] == task]["prag_llm"].to_numpy()[0] if len(df[df["task"] == task]) > 0 else 0 for task in tasks]

    # Filter out NaN values for plotting
    valid_mask = ~(np.isnan(vlm_prag) | np.isnan(llm_prag))
    if not valid_mask.any():
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", fontsize=12)
        ax.set_title("Panel B: VLM vs LLM PRAG", fontweight="bold")
        return

    x_valid = x[valid_mask]
    vlm_valid = np.array(vlm_prag)[valid_mask]
    llm_valid = np.array(llm_prag)[valid_mask]
    tasks_valid = [tasks[i] for i in range(len(tasks)) if valid_mask[i]]

    ax.bar(x_valid - width / 2, vlm_valid, width, label="VLM", color="tab:blue", alpha=0.8)
    ax.bar(x_valid + width / 2, llm_valid, width, label="LLM", color="tab:red", alpha=0.8)

    # Add significance stars if available
    for i, task in enumerate(tasks_valid):
        idx = tasks.index(task)
        if idx < len(df):
            row = df[df["task"] == task]
            if not row.empty and row["test_significant"].to_numpy()[0]:
                p_val = row["test_p_value"].to_numpy()[0]
                star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                ax.text(
                    i,
                    max(vlm_valid[i] if i < len(vlm_valid) else 0, llm_valid[i] if i < len(llm_valid) else 0) + 0.05,
                    star,
                    ha="center",
                    fontsize=12,
                )

    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylabel("PRAG", fontsize=11)
    ax.set_title("Panel B: VLM vs LLM PRAG", fontweight="bold")
    ax.set_xticks(x_valid)
    ax.set_xticklabels(tasks_valid, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0)


def _plot_layerwise_evolution(
    ax: plt.Axes,
    layer_wise_results: dict[str, pd.DataFrame] | None,
    tasks: list[str],  # noqa: ARG001
    target_task: str | None = None,
) -> None:
    """Plot Panel C: Layer-wise進化（3本の曲線）."""
    if layer_wise_results is None or len(layer_wise_results) == 0:
        ax.text(0.5, 0.5, "No layer-wise data", ha="center", va="center", fontsize=12)
        ax.set_title("Panel C: Layer-wise Evolution", fontweight="bold")
        return

    # Use first available task or target_task
    if target_task is None:
        target_task = next(iter(layer_wise_results.keys()))

    if target_task not in layer_wise_results:
        ax.text(0.5, 0.5, f"No data for task: {target_task}", ha="center", va="center", fontsize=12)
        ax.set_title("Panel C: Layer-wise Evolution", fontweight="bold")
        return

    df = layer_wise_results[target_task]

    if df.empty or "layer_num" not in df.columns:
        ax.text(0.5, 0.5, "Invalid layer-wise data", ha="center", va="center", fontsize=12)
        ax.set_title("Panel C: Layer-wise Evolution", fontweight="bold")
        return

    x = df["layer_num"].to_numpy()

    # PRAG (blue)
    if "prag_mean" in df.columns:
        prag = df["prag_mean"].to_numpy()
        ax.plot(x, prag, "b-", label="PRAG", linewidth=2, marker="o", markersize=4)

    # Probe accuracy (green) - should be flat ~0.99
    if "probe_acc" in df.columns:
        probe_acc = df["probe_acc"].to_numpy()
        ax.plot(x, probe_acc, "g-", label="Probe Acc", linewidth=2, marker="s", markersize=4)

    # Decode accuracy (red) - attribute dependent
    if "decode_acc" in df.columns:
        decode_acc = df["decode_acc"].to_numpy()
        ax.plot(x, decode_acc, "r-", label="Decode Acc", linewidth=2, marker="^", markersize=4)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Panel C: Layer-wise Evolution ({target_task})", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)


def _plot_prag_predicts_performance(
    ax: plt.Axes,
    attribute_analysis: pd.DataFrame | None,
) -> None:
    """Plot Panel D: PRAG predicts performance."""
    if attribute_analysis is None or attribute_analysis.empty:
        ax.text(0.5, 0.5, "No attribute analysis data", ha="center", va="center", fontsize=12)
        ax.set_title("Panel D: PRAG Predicts Performance", fontweight="bold")
        return

    if "prag_mean" not in attribute_analysis.columns or "decode_acc" not in attribute_analysis.columns:
        ax.text(0.5, 0.5, "Missing required columns", ha="center", va="center", fontsize=12)
        ax.set_title("Panel D: PRAG Predicts Performance", fontweight="bold")
        return

    # Filter valid data
    valid_mask = ~(np.isnan(attribute_analysis["prag_mean"]) | np.isnan(attribute_analysis["decode_acc"]))
    df_valid = attribute_analysis[valid_mask]

    if df_valid.empty:
        ax.text(0.5, 0.5, "No valid data points", ha="center", va="center", fontsize=12)
        ax.set_title("Panel D: PRAG Predicts Performance", fontweight="bold")
        return

    x = df_valid["prag_mean"].to_numpy()
    y = df_valid["decode_acc"].to_numpy()
    tasks = df_valid["attribute"].to_numpy() if "attribute" in df_valid.columns else None

    # Scatter plot with task colors
    for i, (px, py) in enumerate(zip(x, y, strict=False)):
        color = TASK_COLORS.get(tasks[i] if tasks is not None else f"task_{i}", "tab:blue")
        ax.scatter(px, py, c=color, s=100, alpha=0.7, edgecolors="black", linewidth=1.5)
        if tasks is not None:
            ax.annotate(tasks[i], (px, py), xytext=(5, 5), textcoords="offset points", fontsize=9)

    # Regression line
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2, label="Regression")

        # Calculate correlation
        r = np.corrcoef(x, y)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"r = {r:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

    ax.set_xlabel("PRAG", fontsize=11)
    ax.set_ylabel("Decode Accuracy", fontsize=11)
    ax.set_title("Panel D: PRAG Predicts Performance", fontweight="bold")
    ax.grid(True, alpha=0.3)
    if len(x) > 1:
        ax.legend()


def plot_prag_figure2(
    readout_intervention_results: dict[str, Any] | None = None,
    output_path: Path | None = None,
    figsize: tuple[float, float] = (14, 8),
) -> None:
    """
    Create Figure 2: Causal Intervention.

    Panel A: Readout Replacement実験結果
    Panel B: メカニズム図解

    Args:
        readout_intervention_results: Results from readout intervention experiment
        output_path: Path to save figure
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

    # Panel A: Readout Replacement実験結果
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_readout_intervention_results(ax_a, readout_intervention_results)

    # Panel B: メカニズム図解
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_mechanism_diagram(ax_b)

    plt.suptitle("Figure 2: Causal Intervention", fontsize=16, fontweight="bold", y=0.98)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved Figure 2 to: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def _plot_readout_intervention_results(
    ax: plt.Axes,
    readout_intervention_results: dict[str, Any] | None,
) -> None:
    """Plot Panel A: Readout Replacement実験結果."""
    if readout_intervention_results is None:
        ax.text(0.5, 0.5, "No intervention data", ha="center", va="center", fontsize=12)
        ax.set_title("Panel A: Readout Replacement", fontweight="bold")
        return

    # Extract results
    baseline_acc = readout_intervention_results.get("baseline_acc", 0.0)
    intervention_acc = readout_intervention_results.get("intervention_acc", 0.0)
    improvement = readout_intervention_results.get("improvement", 0.0)

    # Bar chart
    x = np.arange(2)
    accs = [baseline_acc, intervention_acc]
    colors = ["tab:blue", "tab:green"]

    bars = ax.bar(x, accs, color=colors, alpha=0.8, width=0.6)

    # Add value labels
    for _i, (bar, acc) in enumerate(zip(bars, accs, strict=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Add improvement annotation
    if improvement > 0:
        ax.annotate(
            f"+{improvement:.3f}",
            xy=(1, intervention_acc),
            xytext=(1, intervention_acc + 0.1),
            arrowprops={"arrowstyle": "->", "color": "green", "lw": 2},
            fontsize=12,
            fontweight="bold",
            color="green",
            ha="center",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline\n(Unembedding)", "Intervention\n(Probe-based)"])
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Panel A: Readout Replacement", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")


def _plot_mechanism_diagram(ax: plt.Axes) -> None:
    """Plot Panel B: メカニズム図解."""
    ax.text(
        0.5,
        0.8,
        "Hidden Space",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.7},
    )
    ax.text(0.5, 0.5, "↓ Probe direction\n(正しい方向)", ha="center", va="center", fontsize=12, color="green", fontweight="bold")
    ax.text(0.5, 0.3, "↓ Unembedding\n(ズレた方向)", ha="center", va="center", fontsize=12, color="red", fontweight="bold")
    ax.text(0.5, 0.1, "→ Output: 失敗", ha="center", va="center", fontsize=12, color="orange", fontweight="bold")

    # Add arrows
    ax.annotate("", xy=(0.5, 0.65), xytext=(0.5, 0.75), arrowprops={"arrowstyle": "->", "lw": 2, "color": "green"})
    ax.annotate("", xy=(0.5, 0.35), xytext=(0.5, 0.45), arrowprops={"arrowstyle": "->", "lw": 2, "color": "red"})
    ax.annotate("", xy=(0.6, 0.15), xytext=(0.5, 0.25), arrowprops={"arrowstyle": "->", "lw": 2, "color": "orange"})

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Panel B: Mechanism Diagram", fontweight="bold")
    ax.axis("off")
