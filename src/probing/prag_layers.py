"""Layer-wise PRAG tracking across model layers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import HuggingFaceDataset
from src.models.base import BaseFeatureExtractor
from src.probing.prag import (
    compute_prag,
    extract_probe_weights,
    get_lm_head_weights,
    get_task_vocab_ids,
)
from src.probing.trainer import train_eval_probe


def track_prag_across_layers(
    extractor: BaseFeatureExtractor,
    dataset: HuggingFaceDataset,
    results_root: Path,
    task: str,
    layer_names: list[str] | None = None,
    max_iter: int = 2000,
    C: float = 1.0,  # noqa: N803
    solver: str = "lbfgs",
) -> pd.DataFrame:
    """
    Track PRAG across multiple layers.

    Args:
        extractor: Feature extractor model
        dataset: Dataset instance
        results_root: Root directory containing task results
        task: Task name
        layer_names: List of layer names to analyze (e.g., ['l00', 'l01', ...])
                    If None, infer from available feature files
        max_iter: Max iterations for LogisticRegression
        C: Inverse regularization strength
        solver: Solver for LogisticRegression

    Returns:
        DataFrame with PRAG, probe accuracy, and decode accuracy for each layer
    """
    task_dir = results_root / task

    if not task_dir.exists():
        error_msg = f"Task directory not found: {task_dir}"
        raise FileNotFoundError(error_msg)

    # Get lm_head weights once
    lm_head_weights = get_lm_head_weights(extractor)  # [vocab_size, hidden_dim]
    tokenizer = extractor.processor.tokenizer  # type: ignore[union-attr]

    # Get class names from dataset
    task_classes = dataset.classes  # Sorted class names

    # Get task vocabulary IDs
    vocab_ids, _class_to_vocab_id = get_task_vocab_ids(tokenizer, task_classes)
    vocab_ids_tensor = torch.tensor(vocab_ids)
    e_task = lm_head_weights[vocab_ids_tensor]  # [num_classes, hidden_dim]

    # Load labels
    labels_path = task_dir / "labels.npy"
    if not labels_path.exists():
        error_msg = f"Labels file not found: {labels_path}"
        raise FileNotFoundError(error_msg)
    labels = np.load(labels_path)

    # Determine layer names if not provided
    if layer_names is None:
        # Find all feature files
        feature_files = sorted(task_dir.glob("features_*.npy"))
        layer_names = [f.stem.replace("features_", "") for f in feature_files]
        # Filter to LLM layers (l00, l01, ...)
        layer_names = [ln for ln in layer_names if ln.startswith("l") and ln[1:].isdigit()]
        layer_names.sort()

    # Load decode accuracy if available
    decode_log_path = task_dir / "decode_log.csv"
    decode_acc_by_layer: dict[str, float] = {}
    if decode_log_path.exists():
        decode_df = pd.read_csv(decode_log_path)
        if "correct" in decode_df.columns:
            # Decode accuracy is same for all layers (it's from final generation)
            overall_decode_acc = float(decode_df["correct"].mean())
            decode_acc_by_layer = dict.fromkeys(layer_names, overall_decode_acc)

    # Load probe accuracy from metrics if available
    metrics_path = task_dir / "metrics.json"
    probe_acc_by_layer: dict[str, float] = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        for ln in layer_names:
            if ln in metrics:
                probe_acc_by_layer[ln] = metrics[ln].get("acc_mean", np.nan)

    results = []

    for layer_name in layer_names:
        features_path = task_dir / f"features_{layer_name}.npy"

        if not features_path.exists():
            print(f"[WARN] Features not found for {layer_name}, skipping")
            continue

        features = np.load(features_path)

        # Extract probe weights
        try:
            probe_weights, probe_metrics = extract_probe_weights(
                features,
                labels,
                max_iter=max_iter,
                C=C,
                solver=solver,
                use_all_data=True,
            )
        except Exception as e:
            print(f"[ERROR] Failed to extract probe weights for {layer_name}: {e}")
            continue

        # Get probe accuracy (train on all data or from metrics)
        probe_acc = probe_metrics.get("train_acc", np.nan)
        if layer_name in probe_acc_by_layer:
            probe_acc = probe_acc_by_layer[layer_name]

        # Ensure probe_weights and e_task have same number of classes
        if probe_weights.shape[0] != e_task.shape[0]:
            min_classes = min(probe_weights.shape[0], e_task.shape[0])
            probe_weights = probe_weights[:min_classes]
            e_task_aligned = e_task[:min_classes]
        else:
            e_task_aligned = e_task

        # Compute PRAG
        prag_result = compute_prag(probe_weights, e_task_aligned)

        # Get decode accuracy
        decode_acc = decode_acc_by_layer.get(layer_name, np.nan)

        # Extract layer number for sorting
        layer_num = int(layer_name[1:]) if layer_name.startswith("l") and layer_name[1:].isdigit() else -1

        results.append(
            {
                "layer": layer_name,
                "layer_num": layer_num,
                "prag_mean": prag_result["prag_mean"],
                "prag_std": prag_result["prag_std"],
                "prag_min": prag_result["prag_min"],
                "prag_max": prag_result["prag_max"],
                "probe_acc": probe_acc,
                "decode_acc": decode_acc,
                "performance_gap": probe_acc - decode_acc if not (np.isnan(probe_acc) or np.isnan(decode_acc)) else np.nan,
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values("layer_num").reset_index(drop=True)

    return df


def track_prag_across_layers_from_features(
    extractor: BaseFeatureExtractor,
    dataset: HuggingFaceDataset,
    features_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    max_iter: int = 2000,
    C: float = 1.0,  # noqa: N803
    solver: str = "lbfgs",
) -> pd.DataFrame:
    """
    Track PRAG across layers from pre-extracted features.

    Args:
        extractor: Feature extractor model
        dataset: Dataset instance
        features_dict: Dictionary mapping layer names to feature arrays
        labels: Labels array
        max_iter: Max iterations for LogisticRegression
        C: Inverse regularization strength
        solver: Solver for LogisticRegression

    Returns:
        DataFrame with PRAG and probe accuracy for each layer
    """
    # Get lm_head weights once
    lm_head_weights = get_lm_head_weights(extractor)  # [vocab_size, hidden_dim]
    tokenizer = extractor.processor.tokenizer  # type: ignore[union-attr]

    # Get class names from dataset
    task_classes = dataset.classes  # Sorted class names

    # Get task vocabulary IDs
    vocab_ids, _class_to_vocab_id = get_task_vocab_ids(tokenizer, task_classes)
    vocab_ids_tensor = torch.tensor(vocab_ids)
    e_task = lm_head_weights[vocab_ids_tensor]  # [num_classes, hidden_dim]

    results = []

    # Sort layer names
    layer_names = sorted(features_dict.keys(), key=lambda x: int(x[1:]) if x.startswith("l") and x[1:].isdigit() else -1)

    for layer_name in layer_names:
        features = features_dict[layer_name]

        # Extract probe weights
        try:
            probe_weights, probe_metrics = extract_probe_weights(
                features,
                labels,
                max_iter=max_iter,
                C=C,
                solver=solver,
                use_all_data=True,
            )
        except Exception as e:
            print(f"[ERROR] Failed to extract probe weights for {layer_name}: {e}")
            continue

        # Get probe accuracy
        probe_acc = probe_metrics.get("train_acc", np.nan)

        # Also compute CV accuracy for consistency
        try:
            cv_metrics = train_eval_probe(features, labels, max_iter=max_iter, C=C, solver=solver)
            probe_acc = cv_metrics.get("acc_mean", probe_acc)
        except Exception as e:
            # Log exception for debugging
            print(f"[WARN] Failed to compute CV accuracy for {layer_name}: {e}")

        # Ensure probe_weights and e_task have same number of classes
        if probe_weights.shape[0] != e_task.shape[0]:
            min_classes = min(probe_weights.shape[0], e_task.shape[0])
            probe_weights = probe_weights[:min_classes]
            e_task_aligned = e_task[:min_classes]
        else:
            e_task_aligned = e_task

        # Compute PRAG
        prag_result = compute_prag(probe_weights, e_task_aligned)

        # Extract layer number for sorting
        layer_num = int(layer_name[1:]) if layer_name.startswith("l") and layer_name[1:].isdigit() else -1

        results.append(
            {
                "layer": layer_name,
                "layer_num": layer_num,
                "prag_mean": prag_result["prag_mean"],
                "prag_std": prag_result["prag_std"],
                "prag_min": prag_result["prag_min"],
                "prag_max": prag_result["prag_max"],
                "probe_acc": probe_acc,
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values("layer_num").reset_index(drop=True)

    return df
