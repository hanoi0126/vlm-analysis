"""PRAG (Probe-Readout Alignment Gap) calculation module."""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F


def compute_prag(
    w_probe: np.ndarray | torch.Tensor,
    e_task: np.ndarray | torch.Tensor,
) -> dict[str, Any]:
    """
    Compute Probe-Readout Alignment Gap (PRAG).

    PRAG measures the cosine similarity between probe directions and unembedding directions.
    Higher PRAG indicates better alignment.

    Args:
        w_probe: Probe weights [num_classes, hidden_dim]
        e_task: Task vocabulary unembedding rows [num_classes, hidden_dim]

    Returns:
        Dictionary containing:
            - prag_per_class: PRAG score for each class [num_classes]
            - prag_mean: Mean PRAG score
            - prag_std: Standard deviation of PRAG scores
            - prag_min: Minimum PRAG score
            - prag_max: Maximum PRAG score
    """
    # Convert to torch tensors if needed
    if isinstance(w_probe, np.ndarray):
        w_probe = torch.from_numpy(w_probe)
    if isinstance(e_task, np.ndarray):
        e_task = torch.from_numpy(e_task)

    # Ensure float32
    w_probe = w_probe.float()
    e_task = e_task.float()

    # L2 normalization
    w_probe_norm = F.normalize(w_probe, dim=1)  # [C, D]
    e_task_norm = F.normalize(e_task, dim=1)  # [C, D]

    # Cosine similarity (dot product of normalized vectors)
    prag_scores = (w_probe_norm * e_task_norm).sum(dim=1)  # [C]

    # Convert to numpy for consistency
    prag_scores_np = prag_scores.detach().cpu().numpy()

    return {
        "prag_per_class": prag_scores_np.tolist(),
        "prag_mean": float(prag_scores_np.mean()),
        "prag_std": float(prag_scores_np.std()),
        "prag_min": float(prag_scores_np.min()),
        "prag_max": float(prag_scores_np.max()),
    }


def get_task_vocab_ids(
    tokenizer: Any,
    task_classes: list[str],
    use_first_token: bool = True,
) -> tuple[list[int], dict[str, int]]:
    """
    Get vocabulary IDs for task classes.

    Args:
        tokenizer: HuggingFace tokenizer
        task_classes: List of class names (e.g., ['0', '45', '90', '135'])
        use_first_token: If True, use only the first token for multi-token classes

    Returns:
        Tuple of:
            - vocab_ids: List of vocabulary IDs corresponding to task_classes [num_classes]
            - class_to_vocab_id: Mapping from class name to vocab ID
    """
    vocab_ids = []
    class_to_vocab_id = {}

    for cls_name in task_classes:
        # Tokenize class name
        token_ids = tokenizer.encode(cls_name, add_special_tokens=False)

        if len(token_ids) == 0:
            # Fallback: use unknown token
            if tokenizer.unk_token_id is not None:
                token_id = tokenizer.unk_token_id
            else:
                token_id = 0
        elif use_first_token:
            # Use first token only
            token_id = token_ids[0]
        else:
            # Use average embedding (not implemented yet)
            # For now, use first token
            token_id = token_ids[0]

        vocab_ids.append(token_id)
        class_to_vocab_id[cls_name] = token_id

    return vocab_ids, class_to_vocab_id


def extract_probe_weights(
    features: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 2000,
    C: float = 1.0,  # noqa: N803
    solver: str = "lbfgs",
    use_all_data: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Extract probe weights by training a linear probe.

    Args:
        features: Feature matrix (N, D)
        labels: Labels (N,)
        max_iter: Max iterations for LogisticRegression
        C: Inverse regularization strength
        solver: Solver for LogisticRegression
        use_all_data: If True, train on all data. If False, use CV and return last fold weights.

    Returns:
        Tuple of:
            - probe_weights: Probe weights [num_classes, hidden_dim]
            - metrics: Dictionary with training metrics
    """
    # Check if we have enough samples per class
    binc = np.bincount(labels)
    min_per_class = int(binc.min()) if len(binc) > 1 else 0
    n_classes = int(labels.max()) + 1

    if min_per_class < 2:
        error_msg = f"Not enough samples per class (min={min_per_class})"
        raise ValueError(error_msg)

    if use_all_data:
        # Train on all data
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        clf = LogisticRegression(max_iter=max_iter, C=C, solver=solver, random_state=0)
        clf.fit(features_scaled, labels)

        # Extract weights: coef_ is [num_classes, hidden_dim] for multi-class
        probe_weights = clf.coef_  # [num_classes, hidden_dim]

        # Compute accuracy on training data
        y_pred = clf.predict(features_scaled)
        train_acc = float(np.mean(y_pred == labels))

        metrics_dict: dict[str, Any] = {
            "train_acc": train_acc,
            "n": int(features.shape[0]),
            "d": int(features.shape[1]),
            "n_classes": n_classes,
        }
    else:
        # Use CV and return last fold weights (for consistency with train_eval_probe)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        folds = list(skf.split(features, labels))
        tr_idx, _ = folds[-1]  # Use last fold

        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[tr_idx])
        x_test = scaler.transform(features)

        clf = LogisticRegression(max_iter=max_iter, C=C, solver=solver, random_state=0)
        clf.fit(x_train, labels[tr_idx])

        probe_weights = clf.coef_  # [num_classes, hidden_dim]

        y_pred = clf.predict(x_test)
        train_acc = float(np.mean(y_pred == labels))

        metrics_dict = {
            "train_acc": train_acc,
            "n": int(features.shape[0]),
            "d": int(features.shape[1]),
            "n_classes": n_classes,
            "note": "weights from last CV fold",
        }

    return probe_weights, metrics_dict


def get_lm_head_weights(extractor: Any) -> torch.Tensor:
    """
    Get language model head weights from feature extractor.

    Args:
        extractor: BaseFeatureExtractor instance

    Returns:
        Unembedding weight matrix [vocab_size, hidden_dim]
    """
    model = extractor.model

    # Try different methods to access lm_head
    lm_head = None

    # Method 1: get_output_embeddings() (Qwen, LLaVA)
    if hasattr(model, "get_output_embeddings"):
        lm_head = model.get_output_embeddings()  # type: ignore[no-untyped-call]

    # Method 2: language_model.get_output_embeddings() (InternVL)
    if lm_head is None and hasattr(model, "language_model"):
        language_model = model.language_model
        if hasattr(language_model, "get_output_embeddings"):
            lm_head = language_model.get_output_embeddings()  # type: ignore[no-untyped-call]

    # Method 3: Direct access to lm_head attribute
    if lm_head is None and hasattr(model, "lm_head"):
        lm_head = model.lm_head

    if lm_head is None:
        error_msg = "Could not find lm_head in model"
        raise ValueError(error_msg)

    # Get weight matrix
    if hasattr(lm_head, "weight"):
        weight = lm_head.weight  # [vocab_size, hidden_dim]
    else:
        error_msg = "lm_head does not have weight attribute"
        raise ValueError(error_msg)

    return weight.detach().cpu()
