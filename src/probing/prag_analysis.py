"""Attribute-wise PRAG analysis."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data import HuggingFaceDataset
from src.models.base import BaseFeatureExtractor
from src.probing.prag import (
    compute_prag,
    extract_probe_weights,
    get_lm_head_weights,
    get_task_vocab_embeddings,
    get_task_vocab_ids,
)
from src.probing.prag_statistics import PRAGStatistics


def analyze_prag_by_attribute(
    extractor: BaseFeatureExtractor,
    results_root: Path,
    attributes: list[str],
    layer_name: str = "l19",
    max_iter: int = 2000,
    C: float = 1.0,  # noqa: N803
    solver: str = "lbfgs",
) -> pd.DataFrame:
    """
    Analyze PRAG for multiple attributes.

    Args:
        extractor: Feature extractor model
        results_root: Root directory containing task results
        attributes: List of attribute names (e.g., ['color', 'shape', 'angle'])
        layer_name: Layer name to analyze (e.g., 'l19')
        max_iter: Max iterations for LogisticRegression
        C: Inverse regularization strength
        solver: Solver for LogisticRegression

    Returns:
        DataFrame with PRAG and performance metrics for each attribute
    """
    results = []

    # Get lm_head weights once
    lm_head_weights = get_lm_head_weights(extractor)  # [vocab_size, hidden_dim]
    tokenizer = extractor.processor.tokenizer  # type: ignore[union-attr]

    for attr in attributes:
        task_dir = results_root / attr

        if not task_dir.exists():
            print(f"[WARN] Task directory not found: {task_dir}")
            continue

        # Load features and labels
        features_path = task_dir / f"features_{layer_name}.npy"
        labels_path = task_dir / "labels.npy"

        if not features_path.exists() or not labels_path.exists():
            print(f"[WARN] Missing files for {attr}: {features_path}, {labels_path}")
            continue

        features = np.load(features_path)
        labels = np.load(labels_path)

        # Load dataset to get class names
        # Try to load from existing results or infer from labels
        # For now, infer from unique labels
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)

        # Try to load decode accuracy
        decode_acc = np.nan
        decode_log_path = task_dir / "decode_log.csv"
        if decode_log_path.exists():
            decode_df = pd.read_csv(decode_log_path)
            if "correct" in decode_df.columns:
                decode_acc = float(decode_df["correct"].mean())

        # Try to load probe accuracy
        probe_acc = np.nan
        metrics_path = task_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
            if layer_name in metrics:
                probe_acc = metrics[layer_name].get("acc_mean", np.nan)

        # Extract probe weights
        try:
            probe_weights, _probe_metrics = extract_probe_weights(
                features,
                labels,
                max_iter=max_iter,
                C=C,
                solver=solver,
                use_all_data=True,
            )
        except Exception as e:
            print(f"[ERROR] Failed to extract probe weights for {attr}: {e}")
            continue

        # Get task vocabulary IDs
        # Infer class names from dataset
        # For now, use string representation of label IDs
        task_classes = [str(int(i)) for i in range(n_classes)]

        # Try to get actual class names from dataset
        # This requires loading the dataset
        try:
            # Use multi-token aware embedding extraction
            e_task, _token_info = get_task_vocab_embeddings(
                tokenizer=tokenizer,
                task_classes=task_classes,
                lm_head_weights=lm_head_weights,
                use_average_embedding=True,
                verbose=False,
            )
        except Exception as e:
            print(f"[ERROR] Failed to get vocab embeddings for {attr}: {e}")
            continue

        # Convert probe_weights to torch tensor if needed
        if isinstance(probe_weights, np.ndarray):
            probe_weights_torch = torch.from_numpy(probe_weights).float()
        else:
            probe_weights_torch = probe_weights.float()  # type: ignore[unreachable]

        # Ensure probe_weights and e_task have same number of classes
        if probe_weights_torch.shape[0] != e_task.shape[0]:
            print(f"[WARN] Class count mismatch for {attr}: probe={probe_weights_torch.shape[0]}, vocab={e_task.shape[0]}")
            # Use minimum
            min_classes = min(probe_weights_torch.shape[0], e_task.shape[0])
            probe_weights_torch = probe_weights_torch[:min_classes]
            e_task = e_task[:min_classes]

        # Compute PRAG
        prag_result = compute_prag(probe_weights_torch, e_task, verbose=False)

        # Calculate performance gap
        performance_gap = probe_acc - decode_acc if not (np.isnan(probe_acc) or np.isnan(decode_acc)) else np.nan

        results.append(
            {
                "attribute": attr,
                "prag_mean": prag_result["prag_mean"],
                "prag_std": prag_result["prag_std"],
                "prag_min": prag_result["prag_min"],
                "prag_max": prag_result["prag_max"],
                "probe_acc": probe_acc,
                "decode_acc": decode_acc,
                "performance_gap": performance_gap,
                "n_classes": n_classes,
                "n_samples": int(features.shape[0]),
            }
        )

    df = pd.DataFrame(results)

    # Correlation analysis
    if len(df) > 2:
        stats_obj = PRAGStatistics()

        # PRAG vs Decode accuracy
        valid_mask = ~(np.isnan(df["prag_mean"]) | np.isnan(df["decode_acc"]))
        if valid_mask.sum() > 2:
            prag_decode_corr = stats_obj.test_prag_predicts_performance(
                df.loc[valid_mask, "prag_mean"].to_numpy(),
                df.loc[valid_mask, "decode_acc"].to_numpy(),
            )
            df.attrs["prag_decode_correlation"] = prag_decode_corr

        # PRAG vs Performance gap
        valid_mask = ~(np.isnan(df["prag_mean"]) | np.isnan(df["performance_gap"]))
        if valid_mask.sum() > 2:
            prag_gap_corr = stats_obj.test_prag_predicts_performance(
                df.loc[valid_mask, "prag_mean"].to_numpy(),
                df.loc[valid_mask, "performance_gap"].to_numpy(),
            )
            df.attrs["prag_gap_correlation"] = prag_gap_corr

    return df


def analyze_prag_with_dataset_classes(
    extractor: BaseFeatureExtractor,
    dataset: HuggingFaceDataset,
    features: np.ndarray,
    labels: np.ndarray,
    layer_name: str = "l19",
    max_iter: int = 2000,
    C: float = 1.0,  # noqa: N803
    solver: str = "lbfgs",
    debug: bool = False,
) -> dict[str, Any]:
    """
    Analyze PRAG for a single task with dataset class information.

    Args:
        extractor: Feature extractor model
        dataset: Dataset instance with class information
        features: Feature matrix (N, D)
        labels: Labels (N,)
        layer_name: Layer name (for logging)
        max_iter: Max iterations for LogisticRegression
        C: Inverse regularization strength
        solver: Solver for LogisticRegression
        debug: If True, print detailed debug information

    Returns:
        Dictionary with PRAG results and metrics
    """
    # Get lm_head weights
    lm_head_weights = get_lm_head_weights(extractor)  # [vocab_size, hidden_dim]
    tokenizer = extractor.processor.tokenizer  # type: ignore[union-attr]

    # Get class names from dataset
    task_classes = dataset.classes  # Sorted class names

    # Extract probe weights
    probe_weights, probe_metrics = extract_probe_weights(
        features,
        labels,
        max_iter=max_iter,
        C=C,
        solver=solver,
        use_all_data=True,
    )

    # Get task vocabulary embeddings (handles multi-token classes)
    # Get task name from dataset if available
    task_name = getattr(dataset, "task", None) or layer_name
    e_task, token_info = get_task_vocab_embeddings(
        tokenizer=tokenizer,
        task_classes=task_classes,
        lm_head_weights=lm_head_weights,
        use_average_embedding=True,  # Use average embedding for multi-token classes
        verbose=debug,  # Print warnings for multi-token classes if debugging
        debug=debug,  # Enable detailed token ID debugging
        attribute=task_name,
    )

    # Convert probe_weights to torch tensor if needed
    if isinstance(probe_weights, np.ndarray):
        probe_weights_torch = torch.from_numpy(probe_weights).float()
    else:
        probe_weights_torch = probe_weights.float()  # type: ignore[unreachable]

    # Ensure probe_weights and e_task have same number of classes
    if probe_weights_torch.shape[0] != e_task.shape[0]:
        min_classes = min(probe_weights_torch.shape[0], e_task.shape[0])
        probe_weights_torch = probe_weights_torch[:min_classes]
        e_task = e_task[:min_classes]
        print(f"[WARN] Class count mismatch: using {min_classes} classes")

    # Compute PRAG with debug info
    prag_result = compute_prag(probe_weights_torch, e_task, verbose=debug, debug=debug)

    # Also get vocab_ids for backward compatibility
    vocab_ids, class_to_vocab_id = get_task_vocab_ids(tokenizer, task_classes)

    return {
        "prag": prag_result,
        "probe_metrics": probe_metrics,
        "task_classes": task_classes,
        "vocab_ids": vocab_ids,
        "class_to_vocab_id": class_to_vocab_id,
        "token_info": token_info,  # Add tokenization info
        "layer_name": layer_name,
    }
