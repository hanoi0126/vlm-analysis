"""Cross-condition probing for Vision-to-Language translation analysis.

This module implements cross-condition probe training and evaluation:
- Train a probe on one condition (e.g., Image ON) and test on both conditions
- Quantify the geometric structure differences between representation spaces
- Calculate cross-condition accuracy gap as a measure of space similarity
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from src.probing.trainer import safe_macro_ovr_auc


def train_cross_condition_probe(  # noqa: PLR0913
    X_train: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    X_test_same: np.ndarray,  # noqa: N803
    X_test_cross: np.ndarray,  # noqa: N803
    y_test: np.ndarray,
    max_iter: int = 2000,
    C: float = 1.0,  # noqa: N803
    solver: str = "lbfgs",
) -> dict:
    """
    Train probe on one condition, test on both conditions.

    Args:
        X_train: Training features from condition A (N_train, D)
        y_train: Training labels (N_train,)
        X_test_same: Test features from condition A (N_test, D)
        X_test_cross: Test features from condition B (N_test, D)
        y_test: Test labels (N_test,)
        max_iter: Max iterations for LogisticRegression
        C: Inverse regularization strength
        solver: Solver for LogisticRegression

    Returns:
        Dictionary with metrics:
        - same_condition_acc: Accuracy on same condition (Train: A, Test: A)
        - cross_condition_acc: Accuracy on cross condition (Train: A, Test: B)
        - acc_gap: Difference between same and cross condition accuracy
        - same_condition_auc: AUC on same condition
        - cross_condition_auc: AUC on cross condition
        - auc_gap: Difference between same and cross condition AUC
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # noqa: N806
    X_test_same_scaled = scaler.transform(X_test_same)  # noqa: N806
    X_test_cross_scaled = scaler.transform(X_test_cross)  # noqa: N806

    # Train probe
    clf = LogisticRegression(max_iter=max_iter, C=C, solver=solver, random_state=0)
    clf.fit(X_train_scaled, y_train)

    # Predict on same condition
    y_pred_same = clf.predict(X_test_same_scaled)
    acc_same = accuracy_score(y_test, y_pred_same)

    # Predict on cross condition
    y_pred_cross = clf.predict(X_test_cross_scaled)
    acc_cross = accuracy_score(y_test, y_pred_cross)

    # Calculate AUC scores
    n_classes = int(y_test.max()) + 1

    # AUC for same condition
    if hasattr(clf, "predict_proba"):
        y_score_same = clf.predict_proba(X_test_same_scaled)
        y_score_cross = clf.predict_proba(X_test_cross_scaled)
    else:
        df_same = clf.decision_function(X_test_same_scaled)
        df_cross = clf.decision_function(X_test_cross_scaled)
        if df_same.ndim == 1:
            df_same = np.vstack([-df_same, df_same]).T
            df_cross = np.vstack([-df_cross, df_cross]).T
        y_score_same = df_same
        y_score_cross = df_cross

    auc_same = safe_macro_ovr_auc(y_test, y_score_same, n_classes)
    auc_cross = safe_macro_ovr_auc(y_test, y_score_cross, n_classes)

    return {
        "same_condition_acc": float(acc_same),
        "cross_condition_acc": float(acc_cross),
        "acc_gap": float(acc_same - acc_cross),
        "same_condition_auc": float(auc_same),
        "cross_condition_auc": float(auc_cross),
        "auc_gap": float(auc_same - auc_cross),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test_same.shape[0]),
        "d": int(X_train.shape[1]),
    }


def cross_condition_probe_all_layers(  # noqa: PLR0913
    features_condA: dict[str, np.ndarray],  # noqa: N803
    features_condB: dict[str, np.ndarray],  # noqa: N803
    labels: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 0,
    max_iter: int = 2000,
    C: float = 1.0,  # noqa: N803
    solver: str = "lbfgs",
) -> dict[str, dict]:
    """
    Run cross-condition probing for all layers.

    Args:
        features_condA: Features from condition A (e.g., Image ON)
                       Dict mapping layer name to feature array (N, D)
        features_condB: Features from condition B (e.g., Image OFF)
                       Dict mapping layer name to feature array (N, D)
        labels: Labels (N,)
        train_ratio: Ratio of training samples
        seed: Random seed for train/test split
        max_iter: Max iterations for LogisticRegression
        C: Inverse regularization strength
        solver: Solver for LogisticRegression

    Returns:
        Dictionary mapping layer name to results dict with two directions:
        - A_to_B: Train on condition A, test on both A and B
        - B_to_A: Train on condition B, test on both B and A
    """
    # Create train/test split
    rng = np.random.default_rng(seed)
    n = len(labels)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_train = int(n * train_ratio)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    results: dict[str, dict] = {}

    # Get common layers
    common_layers = set(features_condA.keys()) & set(features_condB.keys())

    for layer in sorted(common_layers):
        X_A = features_condA[layer]  # noqa: N806
        X_B = features_condB[layer]  # noqa: N806

        # Check dimensions
        if X_A.shape != X_B.shape:
            print(f"[WARN] Layer {layer}: shape mismatch A={X_A.shape} vs B={X_B.shape}")
            continue

        if X_A.shape[0] != len(labels):
            print(f"[WARN] Layer {layer}: features shape {X_A.shape} != labels shape {len(labels)}")
            continue

        # Direction 1: Train on A, test on A and B
        metrics_A_to_B = train_cross_condition_probe(  # noqa: N806
            X_train=X_A[train_idx],
            y_train=labels[train_idx],
            X_test_same=X_A[test_idx],
            X_test_cross=X_B[test_idx],
            y_test=labels[test_idx],
            max_iter=max_iter,
            C=C,
            solver=solver,
        )

        # Direction 2: Train on B, test on B and A
        metrics_B_to_A = train_cross_condition_probe(  # noqa: N806
            X_train=X_B[train_idx],
            y_train=labels[train_idx],
            X_test_same=X_B[test_idx],
            X_test_cross=X_A[test_idx],
            y_test=labels[test_idx],
            max_iter=max_iter,
            C=C,
            solver=solver,
        )

        results[layer] = {
            "A_to_B": metrics_A_to_B,
            "B_to_A": metrics_B_to_A,
        }

    return results


def summarize_cross_condition_results(
    results: dict[str, dict],
    layer_order: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Summarize cross-condition results into arrays for plotting.

    Args:
        results: Results from cross_condition_probe_all_layers
        layer_order: Order of layers for output arrays

    Returns:
        Dictionary with arrays:
        - layers: Layer names (L,)
        - A_same_acc: Accuracy when training on A and testing on A (L,)
        - A_cross_acc: Accuracy when training on A and testing on B (L,)
        - A_gap: Accuracy gap for A→B (L,)
        - B_same_acc: Accuracy when training on B and testing on B (L,)
        - B_cross_acc: Accuracy when training on B and testing on A (L,)
        - B_gap: Accuracy gap for B→A (L,)
    """
    if layer_order is None:
        layer_order = sorted(results.keys())

    n = len(layer_order)
    A_same_acc = np.full(n, np.nan)  # noqa: N806
    A_cross_acc = np.full(n, np.nan)  # noqa: N806
    A_gap = np.full(n, np.nan)  # noqa: N806
    B_same_acc = np.full(n, np.nan)  # noqa: N806
    B_cross_acc = np.full(n, np.nan)  # noqa: N806
    B_gap = np.full(n, np.nan)  # noqa: N806

    for i, layer in enumerate(layer_order):
        if layer not in results:
            continue

        res = results[layer]

        # A→B direction
        if "A_to_B" in res:
            A_same_acc[i] = res["A_to_B"]["same_condition_acc"]
            A_cross_acc[i] = res["A_to_B"]["cross_condition_acc"]
            A_gap[i] = res["A_to_B"]["acc_gap"]

        # B→A direction
        if "B_to_A" in res:
            B_same_acc[i] = res["B_to_A"]["same_condition_acc"]
            B_cross_acc[i] = res["B_to_A"]["cross_condition_acc"]
            B_gap[i] = res["B_to_A"]["acc_gap"]

    return {
        "layers": np.array(layer_order),
        "A_same_acc": A_same_acc,
        "A_cross_acc": A_cross_acc,
        "A_gap": A_gap,
        "B_same_acc": B_same_acc,
        "B_cross_acc": B_cross_acc,
        "B_gap": B_gap,
    }
