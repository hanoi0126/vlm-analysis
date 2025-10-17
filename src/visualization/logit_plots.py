"""Visualization functions for logit analysis."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_logit_data(task_dir: Path, layer_name: str) -> dict:
    """
    Load logit data for a specific task and layer.

    Args:
        task_dir: Task directory path
        layer_name: Layer name (e.g., 'l00', 'l01')

    Returns:
        Dictionary containing logits, labels, and metadata
    """
    data = {}

    # Load choice logits
    choice_logits_path = task_dir / f"logits_choices_{layer_name}.npy"
    if choice_logits_path.exists():
        data["choice_logits"] = np.load(choice_logits_path)

    # Load labels
    labels_path = task_dir / "labels.npy"
    if labels_path.exists():
        data["labels"] = np.load(labels_path)

    # Load choice metadata
    choice_texts_path = task_dir / "choice_texts.json"
    if choice_texts_path.exists():
        with open(choice_texts_path, encoding="utf-8") as f:
            data["choice_texts"] = json.load(f)

    choice_token_ids_path = task_dir / "choice_token_ids.npy"
    if choice_token_ids_path.exists():
        data["choice_token_ids"] = np.load(choice_token_ids_path, allow_pickle=True)

    # Load decode log if available
    decode_log_path = task_dir / "decode_log.csv"
    if decode_log_path.exists():
        data["decode_log"] = pd.read_csv(decode_log_path)

    return data


def plot_logit_heatmap(
    logits_imageon: np.ndarray,
    logits_imageoff: np.ndarray,
    choice_texts: list[list[str]],
    labels: np.ndarray,  # noqa: ARG001
    layer_name: str,
    output_path: Path,
    title: str = "",
) -> None:
    """
    Plot heatmap of logit differences (image ON - image OFF).

    Args:
        logits_imageon: Logits with images (N, num_choices)
        logits_imageoff: Logits without images (N, num_choices)
        choice_texts: Choice text labels
        labels: Ground truth labels
        layer_name: Layer name
        output_path: Output file path
        title: Plot title
    """
    # Compute logit difference
    logit_diff = logits_imageon - logits_imageoff

    # Get unique choices (assuming all samples have same choices)
    if choice_texts and len(choice_texts) > 0:
        unique_choices = choice_texts[0]
    else:
        num_choices = logit_diff.shape[1]
        unique_choices = [f"Choice {i}" for i in range(num_choices)]

    # Average logit difference for each choice
    mean_diff = np.nanmean(logit_diff, axis=0)

    # Create figure
    _fig, ax = plt.subplots(figsize=(10, 6))

    # Plot heatmap
    sns.heatmap(
        mean_diff.reshape(1, -1),
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        xticklabels=unique_choices,
        yticklabels=["Mean Diff"],
        cbar_kws={"label": "Logit Difference (Image ON - OFF)"},
        ax=ax,
    )

    ax.set_title(f"{title}\nLogit Difference Heatmap - {layer_name}")
    ax.set_xlabel("Choice")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_logit_scatter(
    logits_imageon: np.ndarray,
    logits_imageoff: np.ndarray,
    choice_texts: list[list[str]],
    labels: np.ndarray,
    layer_name: str,
    output_path: Path,
    title: str = "",
) -> None:
    """
    Plot scatter plot of image ON vs image OFF logits.

    Args:
        logits_imageon: Logits with images (N, num_choices)
        logits_imageoff: Logits without images (N, num_choices)
        choice_texts: Choice text labels
        labels: Ground truth labels
        layer_name: Layer name
        output_path: Output file path
        title: Plot title
    """
    # Get unique choices
    if choice_texts and len(choice_texts) > 0:
        unique_choices = choice_texts[0]
    else:
        num_choices = logits_imageon.shape[1]
        unique_choices = [f"Choice {i}" for i in range(num_choices)]

    fig, axes = plt.subplots(1, len(unique_choices), figsize=(5 * len(unique_choices), 5))
    if len(unique_choices) == 1:
        axes = [axes]

    for i, (choice, ax) in enumerate(zip(unique_choices, axes, strict=False)):
        # Extract logits for this choice
        logits_on = logits_imageon[:, i]
        logits_off = logits_imageoff[:, i]

        # Filter out inf values
        valid_mask = np.isfinite(logits_on) & np.isfinite(logits_off)
        logits_on = logits_on[valid_mask]
        logits_off = logits_off[valid_mask]
        sample_labels = labels[valid_mask]

        # Color by correctness
        colors = ["green" if lbl == i else "red" for lbl in sample_labels]

        ax.scatter(logits_off, logits_on, c=colors, alpha=0.5, s=20)
        ax.plot([logits_off.min(), logits_off.max()], [logits_off.min(), logits_off.max()], "k--", alpha=0.3)

        ax.set_xlabel("Image OFF Logit")
        ax.set_ylabel("Image ON Logit")
        ax.set_title(f"Choice: {choice}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{title}\nLogit Scatter Plot - {layer_name}", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confidence_distribution(
    logits_imageon: np.ndarray,
    logits_imageoff: np.ndarray,
    labels: np.ndarray,  # noqa: ARG001
    layer_name: str,
    output_path: Path,
    title: str = "",
) -> None:
    """
    Plot distribution of confidence scores (top1 - top2 logit).

    Args:
        logits_imageon: Logits with images (N, num_choices)
        logits_imageoff: Logits without images (N, num_choices)
        labels: Ground truth labels
        layer_name: Layer name
        output_path: Output file path
        title: Plot title
    """

    # Calculate confidence (difference between top1 and top2)
    def compute_confidence(logits: np.ndarray) -> np.ndarray:
        sorted_logits = np.sort(logits, axis=1)
        return sorted_logits[:, -1] - sorted_logits[:, -2]

    conf_imageon = compute_confidence(logits_imageon)
    conf_imageoff = compute_confidence(logits_imageoff)

    # Filter out invalid values
    valid_mask = np.isfinite(conf_imageon) & np.isfinite(conf_imageoff)
    conf_imageon = conf_imageon[valid_mask]
    conf_imageoff = conf_imageoff[valid_mask]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Histograms
    axes[0].hist(conf_imageon, bins=30, alpha=0.5, label="Image ON", color="blue")
    axes[0].hist(conf_imageoff, bins=30, alpha=0.5, label="Image OFF", color="orange")
    axes[0].set_xlabel("Confidence (Top1 - Top2)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Confidence Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Scatter plot
    axes[1].scatter(conf_imageoff, conf_imageon, alpha=0.5, s=20)
    axes[1].plot(
        [conf_imageoff.min(), conf_imageoff.max()],
        [conf_imageoff.min(), conf_imageoff.max()],
        "k--",
        alpha=0.3,
    )
    axes[1].set_xlabel("Confidence (Image OFF)")
    axes[1].set_ylabel("Confidence (Image ON)")
    axes[1].set_title("Confidence Comparison")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Difference distribution
    conf_diff = conf_imageon - conf_imageoff
    axes[2].hist(conf_diff, bins=30, color="purple", alpha=0.7)
    axes[2].axvline(0, color="black", linestyle="--", linewidth=2, alpha=0.5)
    axes[2].set_xlabel("Confidence Difference (ON - OFF)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Confidence Difference Distribution")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"{title}\nConfidence Analysis - {layer_name}", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_layer_ranking_changes(
    results_root: Path,
    task: str,
    layers: list[str],
    suffix_with_img: str = "_imageon",
    suffix_no_img: str = "_imageoff",
    output_path: Path | None = None,
    title: str = "",
) -> None:
    """
    Plot how choice rankings change across layers.

    Args:
        results_root: Results root directory
        task: Task name
        layers: List of layer names to analyze
        suffix_with_img: Suffix for image ON condition
        suffix_no_img: Suffix for image OFF condition
        output_path: Output file path
        title: Plot title
    """
    task_on_dir = results_root / f"{task}{suffix_with_img}"
    task_off_dir = results_root / f"{task}{suffix_no_img}"

    if not task_on_dir.exists() or not task_off_dir.exists():
        print(f"Warning: Task directories not found for {task}")
        return

    # Load labels
    labels_on = np.load(task_on_dir / "labels.npy")

    # Track rankings across layers
    rankings_on = []
    rankings_off = []

    for layer in layers:
        # Load logits
        data_on = load_logit_data(task_on_dir, layer)
        data_off = load_logit_data(task_off_dir, layer)

        if "choice_logits" in data_on and "choice_logits" in data_off:
            # Get rankings (argsort in descending order)
            ranks_on = np.argsort(-data_on["choice_logits"], axis=1)
            ranks_off = np.argsort(-data_off["choice_logits"], axis=1)

            rankings_on.append(ranks_on)
            rankings_off.append(ranks_off)

    if not rankings_on:
        print(f"Warning: No logit data found for {task}")
        return

    # Plot ranking changes for a few samples
    num_samples = min(5, len(labels_on))
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    for sample_idx in range(num_samples):
        # Image ON
        ax_on = axes[0, sample_idx]
        for choice_idx in range(rankings_on[0].shape[1]):
            ranks = [rankings_on[layer_idx][sample_idx, choice_idx] for layer_idx in range(len(layers))]
            ax_on.plot(range(len(layers)), ranks, marker="o", label=f"Choice {choice_idx}")
        ax_on.set_xlabel("Layer")
        ax_on.set_ylabel("Rank")
        ax_on.set_title(f"Sample {sample_idx} (Image ON)")
        ax_on.legend()
        ax_on.grid(True, alpha=0.3)
        ax_on.invert_yaxis()

        # Image OFF
        ax_off = axes[1, sample_idx]
        for choice_idx in range(rankings_off[0].shape[1]):
            ranks = [rankings_off[layer_idx][sample_idx, choice_idx] for layer_idx in range(len(layers))]
            ax_off.plot(range(len(layers)), ranks, marker="o", label=f"Choice {choice_idx}")
        ax_off.set_xlabel("Layer")
        ax_off.set_ylabel("Rank")
        ax_off.set_title(f"Sample {sample_idx} (Image OFF)")
        ax_off.legend()
        ax_off.grid(True, alpha=0.3)
        ax_off.invert_yaxis()

    fig.suptitle(f"{title}\nChoice Ranking Changes Across Layers - {task}", y=1.0)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_mismatch_cases(
    results_root: Path,
    task: str,
    layer_name: str,
    suffix_with_img: str = "_imageon",
    suffix_no_img: str = "_imageoff",
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Identify and analyze cases where predictions differ between image ON and OFF.

    Args:
        results_root: Results root directory
        task: Task name
        layer_name: Layer name to analyze
        suffix_with_img: Suffix for image ON condition
        suffix_no_img: Suffix for image OFF condition
        output_path: Output CSV path

    Returns:
        DataFrame with mismatch analysis
    """
    task_on_dir = results_root / f"{task}{suffix_with_img}"
    task_off_dir = results_root / f"{task}{suffix_no_img}"

    # Load data
    data_on = load_logit_data(task_on_dir, layer_name)
    data_off = load_logit_data(task_off_dir, layer_name)

    if "choice_logits" not in data_on or "choice_logits" not in data_off:
        print(f"Warning: Logit data not found for {task} - {layer_name}")
        return pd.DataFrame()

    # Get predictions
    pred_on = np.argmax(data_on["choice_logits"], axis=1)
    pred_off = np.argmax(data_off["choice_logits"], axis=1)
    labels = data_on["labels"]

    # Identify mismatches
    mismatch_mask = pred_on != pred_off

    # Create DataFrame
    mismatch_data = []
    for idx in np.where(mismatch_mask)[0]:
        row = {
            "sample_idx": int(idx),
            "ground_truth": int(labels[idx]),
            "pred_imageon": int(pred_on[idx]),
            "pred_imageoff": int(pred_off[idx]),
            "correct_imageon": int(pred_on[idx] == labels[idx]),
            "correct_imageoff": int(pred_off[idx] == labels[idx]),
            "logit_diff_imageon": float(data_on["choice_logits"][idx, pred_on[idx]] - data_on["choice_logits"][idx, pred_off[idx]]),
            "logit_diff_imageoff": float(data_off["choice_logits"][idx, pred_off[idx]] - data_off["choice_logits"][idx, pred_on[idx]]),
        }

        # Add choice texts if available
        if "choice_texts" in data_on and idx < len(data_on["choice_texts"]):
            choices = data_on["choice_texts"][idx]
            row["choice_gt"] = choices[labels[idx]]
            row["choice_imageon"] = choices[pred_on[idx]]
            row["choice_imageoff"] = choices[pred_off[idx]]

        mismatch_data.append(row)

    df = pd.DataFrame(mismatch_data)

    if output_path and not df.empty:
        df.to_csv(output_path, index=False)
        print(f"Saved mismatch analysis to {output_path}")

    return df
