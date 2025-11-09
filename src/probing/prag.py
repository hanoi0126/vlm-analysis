"""PRAG (Probe-Readout Alignment Gap) calculation module."""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import unified_collate


def debug_prag_normalization(
    probe_weight: torch.Tensor,
    e_task: torch.Tensor,
) -> torch.Tensor:
    """
    Debug normalization process for PRAG calculation.

    Args:
        probe_weight: Probe weights [num_classes, hidden_dim]
        e_task: Task vocabulary unembedding rows [num_classes, hidden_dim]

    Returns:
        Cosine similarity scores [num_classes]
    """
    print("\n" + "=" * 80)
    print("=== PRAG Normalization Debug ===")
    print("=" * 80)

    print("\n=== Before Normalization ===")
    print(f"probe_weight shape: {probe_weight.shape}")
    print(f"e_task shape: {e_task.shape}")
    print(f"probe_weight norm (per class): {probe_weight.norm(dim=1)}")
    print(f"e_task norm (per class): {e_task.norm(dim=1)}")
    print(f"probe_weight mean norm: {probe_weight.norm(dim=1).mean():.6f}")
    print(f"e_task mean norm: {e_task.norm(dim=1).mean():.6f}")

    # Normalization
    probe_norm = F.normalize(probe_weight, dim=1, eps=1e-8)
    e_norm = F.normalize(e_task, dim=1, eps=1e-8)

    print("\n=== After Normalization ===")
    print(f"probe_norm norm (per class): {probe_norm.norm(dim=1)}")
    print(f"e_norm norm (per class): {e_norm.norm(dim=1)}")
    all_norms_one = torch.allclose(
        probe_norm.norm(dim=1),
        torch.ones(len(probe_norm), device=probe_norm.device),
        atol=1e-6,
    )
    print(f"All probe norms == 1.0? {all_norms_one}")
    all_norms_one_e = torch.allclose(
        e_norm.norm(dim=1),
        torch.ones(len(e_norm), device=e_norm.device),
        atol=1e-6,
    )
    print(f"All e_task norms == 1.0? {all_norms_one_e}")

    # Cosine similarity
    cosine = (probe_norm * e_norm).sum(dim=1)
    print("\n=== Cosine Similarity ===")
    print(f"Per class: {cosine}")
    print(f"Mean: {cosine.mean():.6f}")
    print(f"Std: {cosine.std():.6f}")
    print(f"Min: {cosine.min():.6f}, Max: {cosine.max():.6f}")

    # Random baseline comparison
    random_probe = torch.randn_like(probe_weight)
    random_e = torch.randn_like(e_task)
    random_probe = F.normalize(random_probe, dim=1, eps=1e-8)
    random_e = F.normalize(random_e, dim=1, eps=1e-8)
    random_cosine = (random_probe * random_e).sum(dim=1).mean()

    print("\n=== Random Baseline ===")
    print(f"Random cosine mean: {random_cosine:.6f}")
    print(f"Actual vs Random: {cosine.mean():.6f} vs {random_cosine:.6f}")
    print(f"Improvement over random: {cosine.mean() - random_cosine:.6f}")

    return cosine


def check_device_dtype(
    probe_weight: torch.Tensor,
    e_task: torch.Tensor,
) -> None:
    """
    Check device and dtype consistency.

    Args:
        probe_weight: Probe weights [num_classes, hidden_dim]
        e_task: Task vocabulary unembedding rows [num_classes, hidden_dim]
    """
    print("\n=== Device/Dtype Check ===")
    print(f"probe_weight: device={probe_weight.device}, dtype={probe_weight.dtype}")
    print(f"e_task: device={e_task.device}, dtype={e_task.dtype}")

    # Check for mismatches
    if probe_weight.device != e_task.device:
        print("⚠️  WARNING: Device mismatch!")
        print(f"   probe_weight on {probe_weight.device}, e_task on {e_task.device}")
    else:
        print("✓ Device match")

    if probe_weight.dtype != e_task.dtype:
        print("⚠️  WARNING: Dtype mismatch!")
        print(f"   probe_weight is {probe_weight.dtype}, e_task is {e_task.dtype}")
    else:
        print("✓ Dtype match")


def validate_probe_weights(
    probe_weight: torch.Tensor,
    num_classes: int,
) -> None:
    """
    Validate probe weights.

    Args:
        probe_weight: Probe weights [num_classes, hidden_dim]
        num_classes: Expected number of classes
    """
    print("\n=== Probe Weight Validation ===")
    print(f"Probe weight shape: {probe_weight.shape}")
    print(f"Expected: ({num_classes}, hidden_dim)")

    if probe_weight.shape[0] != num_classes:
        print(f"⚠️  WARNING: Shape mismatch! Expected {num_classes} classes, got {probe_weight.shape[0]}")

    # Zero vector check
    zero_mask = probe_weight.norm(dim=1) < 1e-6
    if zero_mask.any():
        print(f"⚠️  WARNING: {zero_mask.sum().item()} zero vectors detected!")
        print(f"   Zero vector indices: {torch.where(zero_mask)[0].tolist()}")
    else:
        print("✓ No zero vectors detected")

    # Self-similarity (orthogonality check)
    probe_norm = F.normalize(probe_weight, dim=1, eps=1e-8)
    similarity_matrix = torch.matmul(probe_norm, probe_norm.T)
    eye_mask = torch.eye(num_classes, device=probe_weight.device, dtype=torch.bool)
    off_diagonal = similarity_matrix[~eye_mask]

    print("\nProbe self-similarity (off-diagonal):")
    print(f"  Mean: {off_diagonal.mean():.4f}")
    print(f"  Std: {off_diagonal.std():.4f}")
    print(f"  Min: {off_diagonal.min():.4f}, Max: {off_diagonal.max():.4f}")

    # Expect: close to 0 (high orthogonality)
    if off_diagonal.mean().abs() > 0.3:
        print("⚠️  WARNING: Probe vectors are not orthogonal!")
        print(f"   Mean off-diagonal similarity: {off_diagonal.mean():.4f} (expected < 0.3)")
    else:
        print("✓ Probe vectors are reasonably orthogonal")


def debug_token_ids(
    attribute: str,
    tokenizer: Any,
    task_classes: list[str] | None = None,
) -> None:
    """
    Debug token IDs for task vocabulary.

    Args:
        attribute: Attribute/task name
        tokenizer: HuggingFace tokenizer
        task_classes: List of class names (e.g., ['red', 'blue', 'green'])
    """
    print("\n" + "=" * 80)
    print(f"=== Token ID Debug: {attribute} ===")
    print("=" * 80)

    if task_classes is None:
        print("⚠️  No task_classes provided, skipping token ID debug")
        return

    print(f"\nTask classes: {task_classes}")
    print(f"Number of classes: {len(task_classes)}")

    for i, answer in enumerate(task_classes):
        print(f"\n--- Class {i}: '{answer}' ---")

        # Tokenize as single word
        token_ids = tokenizer.encode(answer, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        print("  Single tokenization:")
        print(f"    Tokens: {tokens}")
        print(f"    Token IDs: {token_ids}")
        print(f"    Number of tokens: {len(token_ids)}")

        if len(token_ids) > 1:
            print(f"    ⚠️  Multi-token: {len(token_ids)} tokens")

        # Tokenize with brackets (as might appear in prompts)
        bracketed = f"{{{answer}}}"
        token_ids_br = tokenizer.encode(bracketed, add_special_tokens=False)
        tokens_br = tokenizer.convert_ids_to_tokens(token_ids_br)

        print(f"  Bracketed tokenization ('{{{answer}}}'):")
        print(f"    Tokens: {tokens_br}")
        print(f"    Token IDs: {token_ids_br}")
        print(f"    Number of tokens: {len(token_ids_br)}")

        if len(token_ids_br) > 3:
            print(f"    ⚠️  Bracketed multi-token: {len(token_ids_br)} tokens")

        # Check for special tokens
        if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in token_ids:
            print(f"    ⚠️  Contains UNK token (ID: {tokenizer.unk_token_id})")


def compute_prag(
    w_probe: np.ndarray | torch.Tensor,
    e_task: np.ndarray | torch.Tensor,
    verbose: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Compute Probe-Readout Alignment Gap (PRAG).

    PRAG measures the cosine similarity between probe directions and unembedding directions.
    Higher PRAG indicates better alignment.

    Args:
        w_probe: Probe weights [num_classes, hidden_dim]
        e_task: Task vocabulary unembedding rows [num_classes, hidden_dim]
        verbose: If True, print basic debug information
        debug: If True, print detailed debug information including normalization,
               device/dtype checks, and probe weight validation

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

    # Ensure same device
    if w_probe.device != e_task.device:
        e_task = e_task.to(w_probe.device)

    # Debug: Device/Dtype check
    if debug:
        check_device_dtype(w_probe, e_task)

    # Debug: Probe weight validation
    if debug:
        num_classes = w_probe.shape[0]
        validate_probe_weights(w_probe, num_classes)

    if verbose or debug:
        print("[DEBUG] PRAG computation:")
        print(f"  w_probe.shape: {w_probe.shape}, device: {w_probe.device}, dtype: {w_probe.dtype}")
        print(f"  e_task.shape: {e_task.shape}, device: {e_task.device}, dtype: {e_task.dtype}")
        print(f"  w_probe norm (before): {w_probe.norm(dim=1)}")
        print(f"  e_task norm (before): {e_task.norm(dim=1)}")

    # Debug: Normalization check
    if debug:
        debug_prag_normalization(w_probe, e_task)

    # L2 normalization
    w_probe_norm = F.normalize(w_probe, dim=1, eps=1e-8)  # [C, D]
    e_task_norm = F.normalize(e_task, dim=1, eps=1e-8)  # [C, D]

    if verbose or debug:
        print(f"  w_probe norm (after): {w_probe_norm.norm(dim=1)}")
        print(f"  e_task norm (after): {e_task_norm.norm(dim=1)}")

    # Cosine similarity (dot product of normalized vectors)
    prag_scores = (w_probe_norm * e_task_norm).sum(dim=1)  # [C]

    if verbose or debug:
        print(f"  Cosine similarity per class: {prag_scores}")
        print(f"  Mean PRAG: {prag_scores.mean():.4f}")
        print(f"  Min PRAG: {prag_scores.min():.4f}")
        print(f"  Max PRAG: {prag_scores.max():.4f}")

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


def get_task_vocab_embeddings(
    tokenizer: Any,
    task_classes: list[str],
    lm_head_weights: torch.Tensor,
    use_average_embedding: bool = True,
    verbose: bool = False,
    debug: bool = False,
    attribute: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Get task vocabulary embeddings, handling multi-token classes.

    For multi-token classes, computes average embedding of all tokens.
    This is critical for PRAG calculation accuracy.

    Args:
        tokenizer: HuggingFace tokenizer
        task_classes: List of class names (e.g., ['upper_left', 'lower_right'])
        lm_head_weights: LM head weight matrix [vocab_size, hidden_dim]
        use_average_embedding: If True, use average embedding for multi-token classes.
                              If False, use last token (often more informative).
        verbose: If True, print warnings for multi-token classes
        debug: If True, print detailed token ID debugging information
        attribute: Attribute/task name for debugging (optional)

    Returns:
        Tuple of:
            - e_task: Task vocabulary embeddings [num_classes, hidden_dim]
            - token_info: Dictionary with tokenization info for debugging
    """
    # Debug: Token ID check
    if debug:
        debug_attr = attribute if attribute is not None else "unknown"
        debug_token_ids(debug_attr, tokenizer, task_classes)

    embeddings = []
    token_info: dict[str, Any] = {
        "single_token_classes": [],
        "multi_token_classes": [],
        "class_to_tokens": {},
    }

    for cls_name in task_classes:
        # Tokenize class name
        token_ids = tokenizer.encode(cls_name, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        if len(token_ids) == 0:
            # Fallback: use unknown token
            if tokenizer.unk_token_id is not None:
                token_id = tokenizer.unk_token_id
            else:
                token_id = 0
            embedding = lm_head_weights[token_id]  # [hidden_dim]
            token_info["single_token_classes"].append(cls_name)
            if verbose:
                print(f"[WARN] Empty tokenization for '{cls_name}', using UNK token")
        elif len(token_ids) == 1:
            # Single token: use directly
            embedding = lm_head_weights[token_ids[0]]  # [hidden_dim]
            token_info["single_token_classes"].append(cls_name)
        else:
            # Multi-token: use average embedding or last token
            token_info["multi_token_classes"].append(cls_name)
            token_info["class_to_tokens"][cls_name] = {
                "tokens": tokens,
                "token_ids": token_ids,
            }

            if verbose:
                print(f"[WARN] Multi-token class '{cls_name}' → {tokens} ({len(token_ids)} tokens)")

            if use_average_embedding:
                # Average embedding of all tokens
                token_embeddings = lm_head_weights[token_ids]  # [n_tokens, hidden_dim]
                embedding = token_embeddings.mean(dim=0)  # [hidden_dim]
            else:
                # Use last token (often more informative for compound words)
                embedding = lm_head_weights[token_ids[-1]]  # [hidden_dim]

        embeddings.append(embedding)

    # Stack into tensor [num_classes, hidden_dim]
    e_task = torch.stack(embeddings)

    return e_task, token_info


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


def debug_llm_condition(
    extractor: Any,
    dataset: Any,
    attribute: str,
    layer_name: str = "last",
    max_iter: int = 2000,
    C: float = 1.0,  # noqa: N803
    solver: str = "lbfgs",
    n_samples: int | None = None,
) -> dict[str, Any]:
    """
    LLM条件でのProbe学習を検証.

    Args:
        extractor: Feature extractor model
        dataset: Dataset instance
        attribute: Attribute/task name
        layer_name: Layer name to extract features from
        max_iter: Max iterations for LogisticRegression
        C: Inverse regularization strength
        solver: Solver for LogisticRegression
        n_samples: Number of samples to use (None for all)

    Returns:
        Dictionary with debug information:
            - probe_acc: Probe accuracy
            - features_shape: Shape of extracted features
            - labels_shape: Shape of labels
            - n_classes: Number of classes
    """
    print("\n" + "=" * 80)
    print(f"=== LLM Condition Debug: {attribute} ===")
    print("=" * 80)

    # Extract features and labels for LLM condition (text-only)
    print("\nExtracting features for LLM condition (text-only)...")
    features_list = []
    labels_list = []

    # Limit samples if specified
    dataset_size = len(dataset)
    if n_samples is not None:
        dataset_size = min(n_samples, dataset_size)
        print(f"  Using {dataset_size} samples (out of {len(dataset)} total)")

    dl = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=unified_collate)

    with torch.no_grad():
        sample_count = 0
        for batch in dl:
            if sample_count >= dataset_size:
                break

            # Prepare text-only prompts (description + question)
            texts = []
            for i, meta in enumerate(batch.get("meta", [])):
                desc = meta.get("description", "") if isinstance(meta, dict) else ""
                question = batch["question"][i] if batch.get("question") else ""
                text = f"{desc}\n\n{question}" if desc else question
                texts.append(text)

            # Extract features without images
            output = extractor.forward(
                images=None,
                texts=texts,
                use_image=False,
                decode=False,
            )

            # Get features from specified layer
            if layer_name == "last":
                # Get last layer
                if output.layers:
                    layer_keys = sorted(output.layers.keys())
                    if layer_keys:
                        features = output.layers[layer_keys[-1]]  # [batch, seq, hidden]
                    else:
                        print("⚠️  No layers found in output")
                        continue
                else:
                    print("⚠️  No layers found in output")
                    continue
            elif layer_name in output.layers:
                features = output.layers[layer_name]
            else:
                print(f"⚠️  Layer {layer_name} not found in output")
                continue

            # Extract last token hidden states
            if features.ndim == 3:
                # [batch, seq, hidden] -> [batch, hidden]
                last_hidden = features[:, -1, :]
            else:
                last_hidden = features

            features_list.append(last_hidden.cpu().numpy())

            # Get labels
            if "label_cls" in batch:
                labels_list.extend(batch["label_cls"])
            elif "labels" in batch:
                label_ids = batch["labels"]
                if hasattr(dataset, "id2cls"):
                    labels_list.extend([dataset.id2cls[int(lid)] for lid in label_ids])
                else:
                    labels_list.extend([str(int(lid)) for lid in label_ids])

            sample_count += len(texts)

    if not features_list:
        print("⚠️  No features extracted")
        return {"error": "No features extracted"}

    # Concatenate features
    features = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list)

    print("\nExtracted features:")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Number of classes: {len(np.unique(labels))}")

    # Convert string labels to integer labels
    unique_labels = sorted(np.unique(labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    label_ids = np.array([label_to_id[label] for label in labels])

    # Train probe
    print("\nTraining probe...")
    probe_weights, probe_metrics = extract_probe_weights(
        features,
        label_ids,
        max_iter=max_iter,
        C=C,
        solver=solver,
        use_all_data=True,
    )

    probe_acc = probe_metrics.get("acc_mean", 0.0)

    print("\n✓ Probe training results:")
    print(f"  Probe accuracy: {probe_acc:.4f}")
    print(f"  Probe weights shape: {probe_weights.shape}")

    if probe_acc < 0.5:
        print("  ⚠️  Low probe accuracy - LLM condition may be inappropriate")

    return {
        "probe_acc": probe_acc,
        "features_shape": features.shape,
        "labels_shape": labels.shape,
        "n_classes": len(unique_labels),
        "probe_weights_shape": probe_weights.shape,
        "unique_labels": unique_labels,
    }


def debug_unembedding_extraction(
    extractor: Any,
    attribute: str,
    task_classes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Unembeddingの取得を検証.

    Args:
        extractor: Feature extractor model
        attribute: Attribute/task name
        task_classes: List of class names (e.g., ['red', 'blue', 'green'])

    Returns:
        Dictionary with debug information:
            - unembedding_shape: Shape of unembedding matrix
            - unembedding_dtype: Dtype of unembedding matrix
            - unembedding_device: Device of unembedding matrix
            - token_info: Tokenization info for each class
    """
    print("\n" + "=" * 80)
    print(f"=== Unembedding Extraction Debug: {attribute} ===")
    print("=" * 80)

    if task_classes is None:
        print("⚠️  No task_classes provided, skipping unembedding debug")
        return {"error": "No task_classes provided"}

    # Get unembedding weights
    print("\nExtracting unembedding weights...")
    lm_head_weights = get_lm_head_weights(extractor)

    print("\nUnembedding matrix:")
    print(f"  Shape: {lm_head_weights.shape}")
    print(f"  Dtype: {lm_head_weights.dtype}")
    print(f"  Device: {lm_head_weights.device}")

    # Get tokenizer
    tokenizer = extractor.processor.tokenizer  # type: ignore[union-attr]

    # Check each class
    print(f"\nTokenization info for {len(task_classes)} classes:")
    token_info = {}

    for i, answer in enumerate(task_classes):
        print(f"\n--- Class {i}: '{answer}' ---")

        # Tokenize
        token_ids = tokenizer.encode(answer, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        print(f"  Number of tokens: {len(token_ids)}")

        if len(token_ids) == 1:
            token_id = token_ids[0]
            embedding = lm_head_weights[token_id]
            embedding_norm = embedding.norm().item()

            print("  ✓ Single token")
            print(f"  Embedding norm: {embedding_norm:.4f}")

            token_info[answer] = {
                "token_ids": token_ids,
                "tokens": tokens,
                "is_single_token": True,
                "embedding_norm": embedding_norm,
            }
        else:
            print(f"  ⚠️  Multi-token: {len(token_ids)} tokens")
            # Get embedding for first token as example
            first_token_embedding = lm_head_weights[token_ids[0]]
            first_norm = first_token_embedding.norm().item()

            print(f"  First token embedding norm: {first_norm:.4f}")

            token_info[answer] = {
                "token_ids": token_ids,
                "tokens": tokens,
                "is_single_token": False,
                "n_tokens": len(token_ids),
                "first_token_embedding_norm": first_norm,
            }

    return {
        "unembedding_shape": lm_head_weights.shape,
        "unembedding_dtype": str(lm_head_weights.dtype),
        "unembedding_device": str(lm_head_weights.device),
        "token_info": token_info,
    }
