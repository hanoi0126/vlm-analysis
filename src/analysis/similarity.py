"""Representation space similarity analysis.

This module provides functions to compute various similarity metrics
between representation spaces, including:
- CKA (Centered Kernel Alignment)
- Cosine Similarity
- Procrustes Distance
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from typing_extensions import assert_never

# Constants
EPSILON = 1e-10  # Small value to avoid division by zero


def center_gram(gram_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Center a Gram matrix.

    Args:
        gram_matrix: Gram matrix of shape (N, N)

    Returns:
        Centered Gram matrix of shape (N, N)
    """
    n = gram_matrix.shape[0]
    centering_matrix = np.eye(n) - np.ones((n, n)) / n
    return centering_matrix @ gram_matrix @ centering_matrix


def compute_cka(
    features_x: NDArray[np.float64],
    features_y: NDArray[np.float64],
    kernel: Literal["linear", "rbf"] = "linear",
) -> float:
    """Compute Centered Kernel Alignment (CKA) between two representations.

    CKA measures similarity between two representation spaces.
    - CKA = 1.0 means perfect alignment
    - CKA = 0.0 means orthogonal spaces

    Reference:
        Kornblith et al. (2019). "Similarity of Neural Network Representations
        Revisited." In ICML.

    Args:
        features_x: First representation matrix of shape (N, D1)
        features_y: Second representation matrix of shape (N, D2)
        kernel: Kernel type, either "linear" or "rbf"

    Returns:
        CKA score between 0.0 and 1.0

    Raises:
        ValueError: If features_x and features_y have different number of samples
    """
    if features_x.shape[0] != features_y.shape[0]:
        msg = f"features_x and features_y must have same number of samples: {features_x.shape[0]} != {features_y.shape[0]}"
        raise ValueError(msg)

    # Ensure float64 to prevent overflow in gram matrix computation
    features_x = features_x.astype(np.float64, copy=False)
    features_y = features_y.astype(np.float64, copy=False)

    # Compute Gram matrices
    if kernel == "linear":
        gram_x = features_x @ features_x.T
        gram_y = features_y @ features_y.T
    elif kernel == "rbf":
        # RBF kernel with median heuristic
        # Compute pairwise distances for median heuristic
        dist_x = np.linalg.norm(features_x[:, None] - features_x[None, :], axis=2)
        dist_y = np.linalg.norm(features_y[:, None] - features_y[None, :], axis=2)

        # Use median as gamma (with small epsilon to avoid division by zero)
        gamma_x = 1.0 / (np.median(dist_x[dist_x > 0]) + EPSILON)
        gamma_y = 1.0 / (np.median(dist_y[dist_y > 0]) + EPSILON)

        gram_x = rbf_kernel(features_x, gamma=gamma_x)
        gram_y = rbf_kernel(features_y, gamma=gamma_y)
    else:
        assert_never(kernel)

    # Center Gram matrices
    gram_x_centered = center_gram(gram_x)
    gram_y_centered = center_gram(gram_y)

    # Compute CKA
    hsic = np.trace(gram_x_centered @ gram_y_centered)
    normalization = np.sqrt(np.trace(gram_x_centered @ gram_x_centered) * np.trace(gram_y_centered @ gram_y_centered))

    if normalization < EPSILON:
        return 0.0

    cka = hsic / normalization
    return float(cka)


def compute_cosine_similarity(
    features_x: NDArray[np.float64],
    features_y: NDArray[np.float64],
    aggregation: Literal["mean", "median"] = "mean",
) -> float:
    """Compute cosine similarity between two representation matrices.

    This function computes pairwise cosine similarities between corresponding
    samples in features_x and features_y, then aggregates them.

    Args:
        features_x: First representation matrix of shape (N, D1)
        features_y: Second representation matrix of shape (N, D2)
        aggregation: How to aggregate pairwise similarities, either "mean" or "median"

    Returns:
        Aggregated cosine similarity between -1.0 and 1.0

    Raises:
        ValueError: If features_x and features_y have different number of samples
    """
    if features_x.shape[0] != features_y.shape[0]:
        msg = f"features_x and features_y must have same number of samples: {features_x.shape[0]} != {features_y.shape[0]}"
        raise ValueError(msg)

    # Ensure float64 to prevent overflow
    features_x = features_x.astype(np.float64, copy=False)
    features_y = features_y.astype(np.float64, copy=False)

    # Normalize to unit vectors
    features_x_norm = features_x / (np.linalg.norm(features_x, axis=1, keepdims=True) + EPSILON)
    features_y_norm = features_y / (np.linalg.norm(features_y, axis=1, keepdims=True) + EPSILON)

    # Compute pairwise cosine similarities (diagonal elements only)
    cos_sims = np.sum(features_x_norm * features_y_norm, axis=1)

    if aggregation == "mean":
        return float(np.mean(cos_sims))

    if aggregation == "median":
        return float(np.median(cos_sims))

    assert_never(aggregation)


def compute_procrustes_distance(
    features_x: NDArray[np.float64],
    features_y: NDArray[np.float64],
) -> float:
    """Compute Procrustes distance between two representation matrices.

    Procrustes analysis finds the optimal rotation and scaling to align
    features_y to features_x, then computes the remaining distance. Lower
    values indicate more similar representations.

    If features_x and features_y have different dimensions, they are first
    projected to a common space using PCA.

    Args:
        features_x: First representation matrix of shape (N, D1)
        features_y: Second representation matrix of shape (N, D2)

    Returns:
        Procrustes distance (lower is more similar, >= 0.0)

    Raises:
        ValueError: If features_x and features_y have different number of samples
    """
    if features_x.shape[0] != features_y.shape[0]:
        msg = f"features_x and features_y must have same number of samples: {features_x.shape[0]} != {features_y.shape[0]}"
        raise ValueError(msg)

    # Ensure float64 to prevent overflow
    features_x = features_x.astype(np.float64, copy=False)
    features_y = features_y.astype(np.float64, copy=False)

    # If dimensions differ, project to common space using PCA
    if features_x.shape[1] != features_y.shape[1]:
        min_dim = min(features_x.shape[1], features_y.shape[1])
        min_dim = min(min_dim, features_x.shape[0] - 1)  # PCA cannot exceed n_samples - 1

        if min_dim < 1:
            # Not enough dimensions for PCA
            return float("nan")

        pca_x = PCA(n_components=min_dim, random_state=0)
        pca_y = PCA(n_components=min_dim, random_state=0)
        features_x = pca_x.fit_transform(features_x)
        features_y = pca_y.fit_transform(features_y)

    # Compute Procrustes distance
    try:
        _, _, disparity = procrustes(features_x, features_y)
        return float(disparity)
    except Exception as e:
        print(f"[WARN] Procrustes computation failed: {e}")
        return float("nan")


def compute_similarity_all_layers(
    features_a: dict[str, NDArray[np.float64]],
    features_b: dict[str, NDArray[np.float64]],
    methods: list[str] | None = None,
    layer_order: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute similarity metrics for all layers.

    This function computes multiple similarity metrics between corresponding
    layers in two feature dictionaries (e.g., Image ON vs Image OFF conditions).

    Args:
        features_a: Features from condition A (e.g., Image ON)
            Dictionary mapping layer names to feature arrays of shape (N, D)
        features_b: Features from condition B (e.g., Image OFF)
            Dictionary mapping layer names to feature arrays of shape (N, D)
        methods: List of methods to use. Available: "cka", "cosine", "procrustes"
            Defaults to ["cka", "cosine"]
        layer_order: Order of layers for output. If None, uses sorted intersection
            of keys from features_a and features_b

    Returns:
        Dictionary mapping layer name to similarity metrics dictionary
        Example: {"l00": {"cka": 0.95, "cosine": 0.87}, ...}
    """
    if methods is None:
        methods = ["cka", "cosine"]

    if layer_order is None:
        layer_order = sorted(set(features_a.keys()) & set(features_b.keys()))

    results: dict[str, dict[str, float]] = {}

    for layer in layer_order:
        if layer not in features_a or layer not in features_b:
            continue

        features_x = features_a[layer]
        features_y = features_b[layer]

        if features_x.shape[0] != features_y.shape[0]:
            print(f"[WARN] Layer {layer}: shape mismatch {features_x.shape} vs {features_y.shape}")
            continue

        metrics: dict[str, float] = {}

        if "cka" in methods:
            try:
                metrics["cka"] = compute_cka(features_x, features_y, kernel="linear")
            except Exception as e:
                print(f"[WARN] CKA failed for layer {layer}: {e}")
                metrics["cka"] = float("nan")

        if "cosine" in methods:
            try:
                metrics["cosine"] = compute_cosine_similarity(features_x, features_y)
            except Exception as e:
                print(f"[WARN] Cosine failed for layer {layer}: {e}")
                metrics["cosine"] = float("nan")

        if "procrustes" in methods:
            try:
                metrics["procrustes"] = compute_procrustes_distance(features_x, features_y)
            except Exception as e:
                print(f"[WARN] Procrustes failed for layer {layer}: {e}")
                metrics["procrustes"] = float("nan")

        results[layer] = metrics

    return results
