"""Visualization utilities for probing results."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.typing import ColorType
import numpy as np
from numpy import dtype, ndarray, signedinteger

if TYPE_CHECKING:
    from numpy._core.multiarray import _Array1D
    from numpy._typing._array_like import NDArray
    from numpy._typing._shape import _AnyShape

# Task colors (fixed for consistency)
TASK_COLORS: dict[str, ColorType] = {
    "color": mcolors.TABLEAU_COLORS["tab:blue"],
    "shape": mcolors.TABLEAU_COLORS["tab:orange"],
    "location": mcolors.TABLEAU_COLORS["tab:green"],
    "angle": mcolors.TABLEAU_COLORS["tab:red"],
    "size": mcolors.TABLEAU_COLORS["tab:purple"],
    "count": mcolors.TABLEAU_COLORS["tab:brown"],
    "position": mcolors.TABLEAU_COLORS["tab:pink"],
    "occlusion": mcolors.TABLEAU_COLORS["tab:olive"],
}

TASK_MARKERS: dict[str, str] = {
    "color": "o",
    "shape": "s",
    "location": "D",
    "angle": "^",
    "size": "v",
    "count": "P",
    "position": "X",
    "occlusion": "*",
}


def _find_metrics_path(
    results_root: Path,
    task: str,
    suffix: str,
) -> Path | None:
    """
    Find metrics.json path for a task.

    Args:
        results_root: Results root directory
        task: Task name
        suffix: Directory suffix

    Returns:
        Path to metrics.json if found, None otherwise
    """
    cand: Path = results_root / f"{task}{suffix}" / "metrics.json"
    if cand.exists():
        return cand

    # Fallback: search for task*
    cands: list[Path] = sorted(results_root.glob(pattern=f"{task}*/metrics.json"))
    return cands[0] if cands else None


def _auto_ylim(
    ax: plt.Axes,
    curves: list[np.ndarray],
    clamp01: bool = False,
) -> None:
    """
    Automatically set y-axis limits with padding.

    Args:
        ax: Matplotlib axes
        curves: List of y-value arrays
        clamp01: Clamp to [0, 1] range
    """
    vals: list[Any] = []
    for y in curves:
        if y is None:
            continue  # type: ignore[unreachable]
        yv: ndarray[_AnyShape, dtype[Any]] = np.asarray(y, dtype=float)
        yv: ndarray[_AnyShape, dtype[Any]] = yv[np.isfinite(yv)]
        yv: ndarray[_AnyShape, dtype[Any]] = yv[~np.isnan(yv)]
        if yv.size:
            vals.append(yv)

    if not vals:
        return

    v: NDArray[Any] = np.concatenate(vals)
    lo, hi = float(np.min(v)), float(np.max(v))

    if lo == hi:
        lo -= 0.05
        hi += 0.05

    pad: float = (hi - lo) * 0.08 + 0.02
    lo, hi = lo - pad, hi + pad

    if clamp01:
        lo, hi = max(0.0, lo), min(1.0, hi)

    ax.set_ylim(bottom=lo, top=hi)
    ax.margins(y=0.05)
    ax.yaxis.set_major_formatter(formatter=FormatStrFormatter(fmt="%.2f"))


def _infer_key_order(results_root: Path, tasks: list[str], suffix: str) -> list[str]:
    """
    Infer key order from available metrics files.

    Args:
        results_root: Results root directory
        tasks: List of task names
        suffix: Directory suffix

    Returns:
        Inferred key order
    """
    for task in tasks:
        mpath = _find_metrics_path(Path(results_root), task, suffix)
        if mpath is not None:
            metrics = json.loads(Path(mpath).read_text(encoding="utf-8"))
            # Extract layer keys and sort them
            keys = list(metrics.keys())
            # Separate v_enc/v_proj and layer keys
            vision_keys = [k for k in keys if k in ["v_enc", "v_proj"]]
            layer_keys = sorted([k for k in keys if k.startswith("l") and k[:3].replace("l", "").isdigit()])
            return vision_keys + layer_keys
    # Fallback to a reasonable default (up to 40 layers should cover most models)
    return ["v_enc", "v_proj"] + [f"l{i:02d}" for i in range(40)]


def plot_probe_curves_multi(
    results_root: Path,
    tasks: list[str],
    suffix: str = "_qwen3b_llmtap",
    key_order: list[str] | None = None,
    title_suffix: str = "",
    show_legend: bool = True,
) -> None:
    """
    Plot probing curves for multiple tasks.

    Args:
        results_root: Results root directory
        tasks: List of task names
        suffix: Directory suffix
        key_order: Order of tap points (default: auto-detected from metrics files)
        title_suffix: Additional title text
        show_legend: Show legend
    """
    if key_order is None:
        key_order = _infer_key_order(results_root, tasks, suffix)

    x: _Array1D[signedinteger[Any]] = np.arange(len(key_order))

    fig_acc, ax_acc = plt.subplots(figsize=(12, 6))
    fig_auc, ax_auc = plt.subplots(figsize=(12, 6))

    acc_curves: list[np.ndarray] = []
    auc_curves: list[np.ndarray] = []
    missing: list[str] = []

    for task in tasks:
        mpath: Path | None = _find_metrics_path(results_root=Path(results_root), task=task, suffix=suffix)
        if mpath is None:
            missing.append(task)
            continue

        metrics = json.loads(s=Path(mpath).read_text(encoding="utf-8"))

        y_acc: NDArray[Any] = np.array(
            [metrics[k]["acc_mean"] if k in metrics else np.nan for k in key_order],
            dtype=float,
        )
        y_auc: NDArray[Any] = np.array(
            [metrics[k]["auc_mean"] if k in metrics else np.nan for k in key_order],
            dtype=float,
        )

        color: ColorType | None = TASK_COLORS.get(task)
        marker: str = TASK_MARKERS.get(task, "o")

        ax_acc.plot(x, y_acc, marker=marker, linestyle="-", label=task, color=color)
        ax_auc.plot(x, y_auc, marker=marker, linestyle="-", label=task, color=color)

        acc_curves.append(y_acc)
        auc_curves.append(y_auc)

    # Format axes
    for ax, what in [(ax_acc, "Accuracy (mean)"), (ax_auc, "ROC-AUC (mean)")]:
        ax.set_xticks(ticks=x)
        ax.set_xticklabels(labels=key_order, rotation=90)
        ax.grid(visible=True, linestyle="--", alpha=0.6)
        if show_legend:
            ax.legend(title="task")
        ax.set_title(label=f"Probing — {what}" + (f" — {title_suffix}" if title_suffix else ""))

    _auto_ylim(ax=ax_acc, curves=acc_curves, clamp01=True)
    _auto_ylim(ax=ax_auc, curves=auc_curves, clamp01=True)

    fig_acc.tight_layout()
    fig_auc.tight_layout()
    plt.show()

    if missing:
        print("[SKIP] metrics.json not found for:", ", ".join(missing))


def plot_comparison(
    results_root: Path,
    tasks: list[str],
    suffix_with_img: str = "_qwen3b_llmtap",
    suffix_no_img: str = "_qwen3b_llmtap_noimage",
    key_order: list[str] | None = None,
    title_suffix: str = "",
) -> None:
    """
    Plot comparison between image-on and image-off experiments.

    Args:
        results_root: Results root directory
        tasks: List of task names
        suffix_with_img: Directory suffix for image-on results
        suffix_no_img: Directory suffix for image-off results
        key_order: Order of tap points (default: auto-detected from metrics files)
        title_suffix: Additional title text
    """
    if key_order is None:
        # Try to infer from either suffix
        key_order = _infer_key_order(results_root, tasks, suffix_with_img)
        if key_order == ["v_enc", "v_proj"] + [f"l{i:02d}" for i in range(40)]:
            # If fallback was used, try the other suffix
            key_order = _infer_key_order(results_root, tasks, suffix_no_img)

    x = np.arange(len(key_order))

    n_tasks = len(tasks)
    fig_acc, axes_acc = plt.subplots(n_tasks, 1, figsize=(14, 4 * n_tasks), squeeze=False)
    fig_auc, axes_auc = plt.subplots(n_tasks, 1, figsize=(14, 4 * n_tasks), squeeze=False)

    for idx, task in enumerate(tasks):
        ax_acc = axes_acc[idx, 0]
        ax_auc = axes_auc[idx, 0]

        # Load metrics for image-on
        mpath_img = _find_metrics_path(Path(results_root), task, suffix_with_img)
        # Load metrics for image-off
        mpath_noimg = _find_metrics_path(Path(results_root), task, suffix_no_img)

        if mpath_img is None and mpath_noimg is None:
            ax_acc.text(0.5, 0.5, f"No data for {task}", ha="center", va="center")
            ax_auc.text(0.5, 0.5, f"No data for {task}", ha="center", va="center")
            continue

        acc_curves = []
        auc_curves = []

        # Plot image-on
        if mpath_img is not None:
            metrics_img = json.loads(Path(mpath_img).read_text(encoding="utf-8"))
            y_acc_img = np.array(
                [metrics_img[k]["acc_mean"] if k in metrics_img else np.nan for k in key_order],
                dtype=float,
            )
            y_auc_img = np.array(
                [metrics_img[k]["auc_mean"] if k in metrics_img else np.nan for k in key_order],
                dtype=float,
            )

            color = TASK_COLORS.get(task, "tab:blue")
            ax_acc.plot(
                x,
                y_acc_img,
                marker="o",
                linestyle="-",
                label="Image ON",
                color=color,
                linewidth=2,
            )
            ax_auc.plot(
                x,
                y_auc_img,
                marker="o",
                linestyle="-",
                label="Image ON",
                color=color,
                linewidth=2,
            )
            acc_curves.append(y_acc_img)
            auc_curves.append(y_auc_img)

        # Plot image-off
        if mpath_noimg is not None:
            metrics_noimg = json.loads(Path(mpath_noimg).read_text(encoding="utf-8"))
            y_acc_noimg = np.array(
                [metrics_noimg[k]["acc_mean"] if k in metrics_noimg else np.nan for k in key_order],
                dtype=float,
            )
            y_auc_noimg = np.array(
                [metrics_noimg[k]["auc_mean"] if k in metrics_noimg else np.nan for k in key_order],
                dtype=float,
            )

            ax_acc.plot(
                x,
                y_acc_noimg,
                marker="s",
                linestyle="--",
                label="Image OFF (text-only)",
                color="gray",
                linewidth=2,
            )
            ax_auc.plot(
                x,
                y_auc_noimg,
                marker="s",
                linestyle="--",
                label="Image OFF (text-only)",
                color="gray",
                linewidth=2,
            )
            acc_curves.append(y_acc_noimg)
            auc_curves.append(y_auc_noimg)

        # Format axes
        for ax, what in [(ax_acc, "Accuracy"), (ax_auc, "ROC-AUC")]:
            ax.set_xticks(x)
            ax.set_xticklabels(key_order, rotation=90, fontsize=8)
            ax.grid(visible=True, linestyle="--", alpha=0.6)
            ax.legend(loc="best")
            title = f"{task} — {what}"
            if title_suffix:
                title += f" — {title_suffix}"
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Layer", fontsize=10)
            ax.set_ylabel(what, fontsize=10)

        _auto_ylim(ax_acc, acc_curves, clamp01=True)
        _auto_ylim(ax_auc, auc_curves, clamp01=True)

    fig_acc.tight_layout()
    fig_auc.tight_layout()
    plt.show()

    print(f"Comparison plots generated for tasks: {', '.join(tasks)}")


def plot_cross_condition_gaps(
    layers: np.ndarray,
    gap_A_to_B: np.ndarray,  # noqa: N803
    gap_B_to_A: np.ndarray,  # noqa: N803
    task: str,
    title_suffix: str = "",
    figsize: tuple[float, float] = (14, 6),
    output_path: str | None = None,
) -> None:
    """
    Plot cross-condition accuracy gaps across layers.

    Args:
        layers: Layer names (L,)
        gap_A_to_B: Accuracy gap when training on A and testing on B (L,)
        gap_B_to_A: Accuracy gap when training on B and testing on A (L,)
        task: Task name
        title_suffix: Additional title text
        figsize: Figure size
        output_path: Path to save the plot (if None, only display)
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(layers))

    # Plot gaps
    color_A = TASK_COLORS.get(task, "tab:blue")  # noqa: N806
    ax.plot(
        x,
        gap_A_to_B,
        marker="o",
        linestyle="-",
        label="ImageON → ImageOFF",
        color=color_A,
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        x,
        gap_B_to_A,
        marker="s",
        linestyle="--",
        label="ImageOFF → ImageON",
        color="gray",
        linewidth=2,
        markersize=6,
    )

    # Add zero line
    ax.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.5)

    # Format axes
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=90, fontsize=8)
    ax.grid(visible=True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")

    title = f"Cross-condition Accuracy Gap — {task}"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Accuracy Gap (same - cross)", fontsize=10)

    # Set y-axis formatter
    ax.yaxis.set_major_formatter(formatter=FormatStrFormatter(fmt="%.2f"))

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    plt.show()
    plt.close(fig)


def plot_cross_condition_prober_accuracy(
    layers: np.ndarray,
    A_same_acc: np.ndarray,  # noqa: N803
    A_cross_acc: np.ndarray,  # noqa: N803
    B_cross_acc: np.ndarray,  # noqa: N803
    B_same_acc: np.ndarray,  # noqa: N803
    task: str,
    title_suffix: str = "",
    figsize: tuple[float, float] = (14, 6),
    output_path: str | None = None,
) -> None:
    """
    Plot cross-condition accuracies for Image and Text probers.

    Args:
        layers: Layer names (L,)
        A_same_acc: Accuracy when training on Image ON and testing on Image ON (L,)
        A_cross_acc: Accuracy when training on Image ON and testing on Image OFF (L,)
        B_cross_acc: Accuracy when training on Image OFF and testing on Image ON (L,)
        B_same_acc: Accuracy when training on Image OFF and testing on Image OFF (L,)
        task: Task name
        title_suffix: Additional title text
        figsize: Figure size
        output_path: Path to save the plot (if None, only display)
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(layers))

    # Color scheme: ImageProber (Image ON) and TextProber (Image OFF)
    color_image = TASK_COLORS.get(task, "tab:blue")
    color_text = "gray"

    # Plot ImageProber lines (trained on Image ON)
    # Same condition: solid line, Cross condition: dashed line
    ax.plot(
        x,
        A_same_acc,
        marker="o",
        linestyle="-",
        label="ImageProber → ImageON",
        color=color_image,
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        x,
        A_cross_acc,
        marker="o",
        linestyle="--",
        label="ImageProber → ImageOFF",
        color=color_image,
        linewidth=2,
        markersize=6,
    )

    # Plot TextProber lines (trained on Image OFF)
    # Cross condition: dashed line, Same condition: solid line
    ax.plot(
        x,
        B_cross_acc,
        marker="s",
        linestyle="--",
        label="TextProber → ImageON",
        color=color_text,
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        x,
        B_same_acc,
        marker="s",
        linestyle="-",
        label="TextProber → ImageOFF",
        color=color_text,
        linewidth=2,
        markersize=6,
    )

    # Format axes
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=90, fontsize=8)
    ax.grid(visible=True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")

    title = f"Cross-condition Accuracy — {task}"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)

    # Set y-axis formatter
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    plt.show()
    plt.close(fig)


def plot_cross_condition_matrix(
    task: str,
    layer: str,
    metrics: dict,
    figsize: tuple[float, float] = (10, 5),
    output_path: str | None = None,
) -> None:
    """
    Plot cross-condition accuracy matrix for a specific task and layer.

    Shows a 2x2 matrix:
             Test: ImageON  |  Test: ImageOFF
    Train: ImageON      acc_same    |    acc_cross
    Train: ImageOFF     acc_cross   |    acc_same

    Args:
        task: Task name
        layer: Layer name
        metrics: Metrics dict with keys "A_to_B" and "B_to_A"
        figsize: Figure size
        output_path: Path to save the plot (if None, only display)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Extract metrics
    A_to_B = metrics.get("A_to_B", {})  # noqa: N806
    B_to_A = metrics.get("B_to_A", {})  # noqa: N806

    # Matrix 1: Accuracy
    acc_matrix = np.array(
        [
            [A_to_B.get("same_condition_acc", np.nan), A_to_B.get("cross_condition_acc", np.nan)],
            [B_to_A.get("cross_condition_acc", np.nan), B_to_A.get("same_condition_acc", np.nan)],
        ]
    )

    # Matrix 2: AUC
    auc_matrix = np.array(
        [
            [A_to_B.get("same_condition_auc", np.nan), A_to_B.get("cross_condition_auc", np.nan)],
            [B_to_A.get("cross_condition_auc", np.nan), B_to_A.get("same_condition_auc", np.nan)],
        ]
    )

    # Plot accuracy matrix
    im1 = ax1.imshow(acc_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Test: ImageON", "Test: ImageOFF"])
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Train: ImageON", "Train: ImageOFF"])
    ax1.set_title(f"Accuracy — {task} — {layer}", fontweight="bold")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax1.text(
                j,
                i,
                f"{acc_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=14,
                fontweight="bold",
            )

    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Plot AUC matrix
    im2 = ax2.imshow(auc_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Test: ImageON", "Test: ImageOFF"])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Train: ImageON", "Train: ImageOFF"])
    ax2.set_title(f"AUC — {task} — {layer}", fontweight="bold")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax2.text(
                j,
                i,
                f"{auc_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=14,
                fontweight="bold",
            )

    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    plt.show()
    plt.close(fig)
