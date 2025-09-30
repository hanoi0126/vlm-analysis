"""Visualization utilities for probing results."""

import json
from pathlib import Path
from typing import List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


# Task colors (fixed for consistency)
TASK_COLORS = {
    "color": mcolors.TABLEAU_COLORS["tab:blue"],
    "shape": mcolors.TABLEAU_COLORS["tab:orange"],
    "location": mcolors.TABLEAU_COLORS["tab:green"],
    "angle": mcolors.TABLEAU_COLORS["tab:red"],
    "size": mcolors.TABLEAU_COLORS["tab:purple"],
    "count": mcolors.TABLEAU_COLORS["tab:brown"],
    "position": mcolors.TABLEAU_COLORS["tab:pink"],
    "occlusion": mcolors.TABLEAU_COLORS["tab:gray"],
}

TASK_MARKERS = {
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
) -> Optional[Path]:
    """
    Find metrics.json path for a task.

    Args:
        results_root: Results root directory
        task: Task name
        suffix: Directory suffix

    Returns:
        Path to metrics.json if found, None otherwise
    """
    cand = results_root / f"{task}{suffix}" / "metrics.json"
    if cand.exists():
        return cand

    # Fallback: search for task*
    cands = sorted(results_root.glob(f"{task}*/metrics.json"))
    return cands[0] if cands else None


def _auto_ylim(
    ax: plt.Axes,
    curves: List[np.ndarray],
    clamp01: bool = False,
) -> None:
    """
    Automatically set y-axis limits with padding.

    Args:
        ax: Matplotlib axes
        curves: List of y-value arrays
        clamp01: Clamp to [0, 1] range
    """
    vals = []
    for y in curves:
        if y is None:
            continue
        yv = np.asarray(y, dtype=float)
        yv = yv[np.isfinite(yv)]
        yv = yv[~np.isnan(yv)]
        if yv.size:
            vals.append(yv)

    if not vals:
        return

    v = np.concatenate(vals)
    lo, hi = float(np.min(v)), float(np.max(v))

    if lo == hi:
        lo -= 0.05
        hi += 0.05

    pad = (hi - lo) * 0.08 + 0.02
    lo, hi = lo - pad, hi + pad

    if clamp01:
        lo, hi = max(0.0, lo), min(1.0, hi)

    ax.set_ylim(lo, hi)
    ax.margins(y=0.05)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def plot_probe_curves_multi(
    results_root: Path,
    tasks: List[str],
    suffix: str = "_qwen3b_llmtap",
    key_order: Optional[List[str]] = None,
    title_suffix: str = "",
    show_legend: bool = True,
) -> None:
    """
    Plot probing curves for multiple tasks.

    Args:
        results_root: Results root directory
        tasks: List of task names
        suffix: Directory suffix
        key_order: Order of tap points (default: ['pre', 'post', 'l00', ...])
        title_suffix: Additional title text
        show_legend: Show legend
    """
    if key_order is None:
        key_order = ["pre", "post"] + [f"l{i:02d}" for i in range(36)]

    x = np.arange(len(key_order))

    fig_acc, ax_acc = plt.subplots(figsize=(12, 6))
    fig_auc, ax_auc = plt.subplots(figsize=(12, 6))

    acc_curves: List[np.ndarray] = []
    auc_curves: List[np.ndarray] = []
    missing: List[str] = []

    for task in tasks:
        mpath = _find_metrics_path(Path(results_root), task, suffix)
        if mpath is None:
            missing.append(task)
            continue

        metrics = json.loads(Path(mpath).read_text(encoding="utf-8"))

        y_acc = np.array(
            [metrics[k]["acc_mean"] if k in metrics else np.nan for k in key_order],
            dtype=float,
        )
        y_auc = np.array(
            [metrics[k]["auc_mean"] if k in metrics else np.nan for k in key_order],
            dtype=float,
        )

        color = TASK_COLORS.get(task, None)
        marker = TASK_MARKERS.get(task, "o")

        ax_acc.plot(x, y_acc, marker=marker, linestyle="-", label=task, color=color)
        ax_auc.plot(x, y_auc, marker=marker, linestyle="-", label=task, color=color)

        acc_curves.append(y_acc)
        auc_curves.append(y_auc)

    # Format axes
    for ax, what in [(ax_acc, "Accuracy (mean)"), (ax_auc, "ROC-AUC (mean)")]:
        ax.set_xticks(x)
        ax.set_xticklabels(key_order, rotation=90)
        ax.grid(True, linestyle="--", alpha=0.6)
        if show_legend:
            ax.legend(title="task")
        ax.set_title(
            f"Probing — {what}" + (f" — {title_suffix}" if title_suffix else "")
        )

    _auto_ylim(ax_acc, acc_curves, clamp01=True)
    _auto_ylim(ax_auc, auc_curves, clamp01=True)

    fig_acc.tight_layout()
    fig_auc.tight_layout()
    plt.show()

    if missing:
        print("[SKIP] metrics.json not found for:", ", ".join(missing))
