"""Visualization functions for logit analysis."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
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


def plot_choice_probabilities_across_layers(
    results_root: Path,
    task: str,
    layers: list[str],
    suffix_with_img: str = "_imageon",
    suffix_no_img: str = "_imageoff",
    output_path: Path | None = None,
    num_samples: int = 5,
    use_mean: bool = False,
) -> None:
    """
    Plot choice probabilities (from logits) across layers for Image ON vs OFF.

    Args:
        results_root: Results root directory
        task: Task name
        layers: List of layer names to analyze
        suffix_with_img: Suffix for image ON condition
        suffix_no_img: Suffix for image OFF condition
        output_path: Output file path
        num_samples: Number of sample plots to generate (if use_mean=False)
        use_mean: If True, plot mean probabilities across all samples
    """
    task_on_dir = results_root / f"{task}{suffix_with_img}"
    task_off_dir = results_root / f"{task}{suffix_no_img}"

    if not task_on_dir.exists() or not task_off_dir.exists():
        print(f"Warning: Task directories not found for {task}")
        return

    # Load logits for all layers
    logits_on_all = []
    logits_off_all = []
    choice_texts = None
    labels = None

    # Load decode logs for actual predictions
    decode_log_on = None
    decode_log_off = None
    decode_log_on_path = task_on_dir / "decode_log.csv"
    decode_log_off_path = task_off_dir / "decode_log.csv"

    if decode_log_on_path.exists():
        decode_log_on = pd.read_csv(decode_log_on_path)
    if decode_log_off_path.exists():
        decode_log_off = pd.read_csv(decode_log_off_path)

    for layer in layers:
        data_on = load_logit_data(task_on_dir, layer)
        data_off = load_logit_data(task_off_dir, layer)

        if "choice_logits" not in data_on or "choice_logits" not in data_off:
            print(f"Warning: Logit data not found for layer {layer}")
            continue

        logits_on_all.append(data_on["choice_logits"])
        logits_off_all.append(data_off["choice_logits"])

        if choice_texts is None and "choice_texts" in data_on:
            choice_texts = data_on["choice_texts"]
        if labels is None and "labels" in data_on:
            labels = data_on["labels"]

    if not logits_on_all or not logits_off_all:
        print(f"Warning: No logit data found for {task}")
        return

    # Convert to numpy arrays: (num_layers, num_samples, num_choices)
    logits_on_array = np.array(logits_on_all)
    logits_off_array = np.array(logits_off_all)

    # Convert logits to probabilities using softmax
    probs_on_array = softmax(logits_on_array, axis=2)
    probs_off_array = softmax(logits_off_array, axis=2)

    _num_layers, num_samples_total, num_choices = probs_on_array.shape

    # Get choice labels
    if choice_texts and len(choice_texts) > 0:
        choice_labels = choice_texts[0]
    else:
        choice_labels = [f"Choice {i}" for i in range(num_choices)]

    if use_mean:
        # Plot mean probabilities across all samples
        _plot_mean_probabilities(
            probs_on_array,
            probs_off_array,
            layers,
            choice_labels,
            task,
            output_path,
        )
    else:
        # Plot individual samples
        num_samples_to_plot = min(num_samples, num_samples_total)
        _plot_sample_probabilities(
            probs_on_array,
            probs_off_array,
            layers,
            choice_labels,
            labels,
            num_samples_to_plot,
            task,
            output_path,
            decode_log_on,
            decode_log_off,
        )


def _plot_mean_probabilities(
    probs_on: np.ndarray,
    probs_off: np.ndarray,
    layers: list[str],
    choice_labels: list[str],
    task: str,
    output_path: Path | None,
) -> None:
    """Plot mean probabilities across all samples."""
    # Calculate mean across samples: (num_layers, num_choices)
    mean_probs_on = np.mean(probs_on, axis=1)
    mean_probs_off = np.mean(probs_off, axis=1)

    _fig, (ax_on, ax_off) = plt.subplots(1, 2, figsize=(16, 6))

    layer_indices = range(len(layers))

    # Plot Image ON
    for choice_idx, label in enumerate(choice_labels):
        ax_on.plot(
            layer_indices,
            mean_probs_on[:, choice_idx],
            marker="o",
            label=label,
            linewidth=2,
        )

    ax_on.set_xlabel("Layer", fontsize=12)
    ax_on.set_ylabel("Probability", fontsize=12)
    ax_on.set_title(f"{task.upper()} - Image ON (Mean)", fontsize=14, fontweight="bold")
    ax_on.set_xticks(layer_indices[:: max(1, len(layers) // 10)])
    ax_on.set_xticklabels(layers[:: max(1, len(layers) // 10)])
    ax_on.legend(loc="best")
    ax_on.grid(True, alpha=0.3)
    ax_on.set_ylim(0, 1)

    # Plot Image OFF
    for choice_idx, label in enumerate(choice_labels):
        ax_off.plot(
            layer_indices,
            mean_probs_off[:, choice_idx],
            marker="o",
            label=label,
            linewidth=2,
        )

    ax_off.set_xlabel("Layer", fontsize=12)
    ax_off.set_ylabel("Probability", fontsize=12)
    ax_off.set_title(f"{task.upper()} - Image OFF (Mean)", fontsize=14, fontweight="bold")
    ax_off.set_xticks(layer_indices[:: max(1, len(layers) // 10)])
    ax_off.set_xticklabels(layers[:: max(1, len(layers) // 10)])
    ax_off.legend(loc="best")
    ax_off.grid(True, alpha=0.3)
    ax_off.set_ylim(0, 1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _plot_sample_probabilities(
    probs_on: np.ndarray,
    probs_off: np.ndarray,
    layers: list[str],
    choice_labels: list[str],
    labels: np.ndarray | None,
    num_samples: int,
    task: str,
    output_path: Path | None,
    decode_log_on: pd.DataFrame | None = None,
    decode_log_off: pd.DataFrame | None = None,
) -> None:
    """Plot probabilities for individual samples."""
    num_samples_total = probs_on.shape[1]

    # Select samples to plot (evenly distributed)
    sample_indices = np.linspace(0, num_samples_total - 1, num_samples, dtype=int)

    fig, axes = plt.subplots(2, num_samples, figsize=(5 * num_samples, 10))
    if num_samples == 1:
        axes = axes.reshape(2, 1)

    layer_indices = range(len(layers))

    for sample_idx, real_sample_idx in enumerate(sample_indices):
        # Get GT from decode_log (実際のground truth)
        if decode_log_on is not None and real_sample_idx < len(decode_log_on):
            gt_text = decode_log_on.iloc[real_sample_idx]["ground_truth"]
            gt_label = f"GT: {gt_text}"
        elif decode_log_off is not None and real_sample_idx < len(decode_log_off):
            gt_text = decode_log_off.iloc[real_sample_idx]["ground_truth"]
            gt_label = f"GT: {gt_text}"
        elif labels is not None:
            # Fallback to labels array
            gt_label = f"GT: {choice_labels[labels[real_sample_idx]]}"
        else:
            gt_label = "GT: Unknown"

        # Get actual predictions from decode_log (実際の生成結果)
        if decode_log_on is not None and real_sample_idx < len(decode_log_on):
            actual_pred_on = decode_log_on.iloc[real_sample_idx]["gen_parsed"]
            pred_on_label = f"VLM: {actual_pred_on}" if actual_pred_on else "VLM: None"
        else:
            # Fallback to logit-based prediction
            pred_on = np.argmax(probs_on[-1, real_sample_idx, :])
            pred_on_label = f"VLM: {choice_labels[pred_on]}"

        if decode_log_off is not None and real_sample_idx < len(decode_log_off):
            actual_pred_off = decode_log_off.iloc[real_sample_idx]["gen_parsed"]
            pred_off_label = f"LLM: {actual_pred_off}" if actual_pred_off else "LLM: None"
        else:
            # Fallback to logit-based prediction
            pred_off = np.argmax(probs_off[-1, real_sample_idx, :])
            pred_off_label = f"LLM: {choice_labels[pred_off]}"

        # Image ON (top row)
        ax_on = axes[0, sample_idx]
        for choice_idx, label in enumerate(choice_labels):
            ax_on.plot(
                layer_indices,
                probs_on[:, real_sample_idx, choice_idx],
                marker="o",
                label=label,
                linewidth=2,
            )

        ax_on.set_xlabel("Layer")
        ax_on.set_ylabel("Probability")
        title_on = f"Sample {real_sample_idx} (Image ON)\n{gt_label} | {pred_on_label}"
        ax_on.set_title(title_on, fontsize=10)
        ax_on.set_xticks(layer_indices[:: max(1, len(layers) // 5)])
        ax_on.set_xticklabels(layers[:: max(1, len(layers) // 5)], rotation=45, ha="right")
        ax_on.legend(loc="best", fontsize=8)
        ax_on.grid(True, alpha=0.3)
        ax_on.set_ylim(0, 1)

        # Image OFF (bottom row)
        ax_off = axes[1, sample_idx]
        for choice_idx, label in enumerate(choice_labels):
            ax_off.plot(
                layer_indices,
                probs_off[:, real_sample_idx, choice_idx],
                marker="o",
                label=label,
                linewidth=2,
            )

        ax_off.set_xlabel("Layer")
        ax_off.set_ylabel("Probability")
        title_off = f"Sample {real_sample_idx} (Image OFF)\n{gt_label} | {pred_off_label}"
        ax_off.set_title(title_off, fontsize=10)
        ax_off.set_xticks(layer_indices[:: max(1, len(layers) // 5)])
        ax_off.set_xticklabels(layers[:: max(1, len(layers) // 5)], rotation=45, ha="right")
        ax_off.legend(loc="best", fontsize=8)
        ax_off.grid(True, alpha=0.3)
        ax_off.set_ylim(0, 1)

    fig.suptitle(f"{task.upper()} - Choice Probabilities Across Layers", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
