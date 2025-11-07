"""Statistical validation functions for ablation experiments."""

import numpy as np


def bootstrap_confidence_interval(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 1000,
    confidence: float = 0.95,
    metric: str = "accuracy",
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        predictions: Model predictions (shape: [n_samples])
        labels: Ground truth labels (shape: [n_samples])
        n_samples: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        metric: Metric to compute ('accuracy', 'top3', etc.)

    Returns:
        Tuple of (mean_metric, ci_lower, ci_upper)
    """
    n_data = len(predictions)
    bootstrap_metrics = []

    rng = np.random.RandomState(42)

    for _ in range(n_samples):
        # Resample with replacement
        indices = rng.choice(n_data, size=n_data, replace=True)
        sample_preds = predictions[indices]
        sample_labels = labels[indices]

        # Compute metric
        if metric == "accuracy":
            sample_metric = np.mean(sample_preds == sample_labels)
        elif metric == "top3":
            # Assumes predictions are indices for top-3
            sample_metric = np.mean([pred in label for pred, label in zip(sample_preds, sample_labels, strict=False)])
        else:
            msg = f"Unknown metric: {metric}"
            raise ValueError(msg)

        bootstrap_metrics.append(sample_metric)

    bootstrap_metrics = np.array(bootstrap_metrics)

    # Compute confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))
    mean_metric = np.mean(bootstrap_metrics)

    return mean_metric, ci_lower, ci_upper


def permutation_test(
    baseline_preds: np.ndarray,
    ablated_preds: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 1000,
) -> float:
    """
    Perform permutation test to assess significance of ablation effect.

    Tests null hypothesis: ablation has no effect on prediction accuracy.

    Args:
        baseline_preds: Predictions without ablation
        ablated_preds: Predictions with ablation
        labels: Ground truth labels
        n_permutations: Number of permutations

    Returns:
        p-value for the test
    """
    # Compute observed difference
    baseline_acc = np.mean(baseline_preds == labels)
    ablated_acc = np.mean(ablated_preds == labels)
    observed_diff = baseline_acc - ablated_acc

    # Permutation test
    n_data = len(labels)
    rng = np.random.RandomState(42)

    # Combine predictions
    combined = np.stack([baseline_preds, ablated_preds], axis=1)

    count_extreme = 0
    for _ in range(n_permutations):
        # Randomly swap between baseline and ablated for each sample
        swaps = rng.randint(0, 2, size=n_data)
        perm_baseline = combined[np.arange(n_data), swaps]
        perm_ablated = combined[np.arange(n_data), 1 - swaps]

        # Compute permuted difference
        perm_baseline_acc = np.mean(perm_baseline == labels)
        perm_ablated_acc = np.mean(perm_ablated == labels)
        perm_diff = perm_baseline_acc - perm_ablated_acc

        # Count how often permuted difference is as extreme as observed
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)
    return p_value


def multiple_comparison_correction(
    p_values: np.ndarray | list[float],
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply multiple comparison correction to p-values.

    Args:
        p_values: Array of p-values
        method: Correction method ('bonferroni', 'fdr', 'holm')
        alpha: Significance level

    Returns:
        Tuple of (corrected_p_values, is_significant)
    """
    p_values = np.asarray(p_values)
    n_tests = len(p_values)

    if method == "bonferroni":
        # Bonferroni correction
        corrected_p = p_values * n_tests
        corrected_p = np.minimum(corrected_p, 1.0)
        is_significant = corrected_p < alpha

    elif method == "holm":
        # Holm-Bonferroni method (sequential)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        corrected_p = np.zeros_like(p_values)
        is_significant = np.zeros(n_tests, dtype=bool)

        for i, idx in enumerate(sorted_indices):
            corrected_p[idx] = sorted_p[i] * (n_tests - i)
            corrected_p[idx] = min(corrected_p[idx], 1.0)
            is_significant[idx] = corrected_p[idx] < alpha

    elif method == "fdr":
        # Benjamini-Hochberg FDR control
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        # Compute adjusted p-values
        adjusted_p = np.zeros_like(p_values)
        for i, idx in enumerate(sorted_indices):
            adjusted_p[idx] = sorted_p[i] * n_tests / (i + 1)

        # Enforce monotonicity
        for i in range(n_tests - 1, 0, -1):
            idx_current = sorted_indices[i]
            idx_prev = sorted_indices[i - 1]
            adjusted_p[idx_prev] = min(adjusted_p[idx_prev], adjusted_p[idx_current])

        adjusted_p = np.minimum(adjusted_p, 1.0)
        corrected_p = adjusted_p
        is_significant = corrected_p < alpha

    else:
        msg = f"Unknown correction method: {method}"
        raise ValueError(msg)

    return corrected_p, is_significant


def compute_effect_size(
    baseline_acc: float,
    ablated_acc: float,
    baseline_preds: np.ndarray | None = None,
    ablated_preds: np.ndarray | None = None,
) -> float:
    """
    Compute effect size (Cohen's d) for ablation experiment.

    Args:
        baseline_acc: Baseline accuracy
        ablated_acc: Ablated accuracy
        baseline_preds: Optional baseline predictions for variance estimation
        ablated_preds: Optional ablated predictions for variance estimation

    Returns:
        Cohen's d effect size
    """
    # Simple effect size based on difference
    if baseline_preds is None or ablated_preds is None:
        # Use pooled variance estimate based on binomial distribution
        # For binary accuracy: variance = p(1-p)
        var_baseline = baseline_acc * (1 - baseline_acc)
        var_ablated = ablated_acc * (1 - ablated_acc)
        pooled_std = np.sqrt((var_baseline + var_ablated) / 2)

        if pooled_std < 1e-10:
            return 0.0

        cohens_d = (baseline_acc - ablated_acc) / pooled_std
    else:
        # Compute from actual predictions
        baseline_correct = baseline_preds.astype(float)
        ablated_correct = ablated_preds.astype(float)

        std_baseline = np.std(baseline_correct)
        std_ablated = np.std(ablated_correct)
        pooled_std = np.sqrt((std_baseline**2 + std_ablated**2) / 2)

        if pooled_std < 1e-10:
            return 0.0

        cohens_d = (baseline_acc - ablated_acc) / pooled_std

    return cohens_d


def compute_prediction_entropy(probabilities: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy of prediction probabilities.

    High entropy = uncertain predictions.

    Args:
        probabilities: Probability distributions (shape: [n_samples, n_classes])

    Returns:
        Entropy values (shape: [n_samples])
    """
    # Avoid log(0)
    probabilities = np.clip(probabilities, 1e-10, 1.0)

    # Shannon entropy: H = -sum(p * log(p))
    entropy = -np.sum(probabilities * np.log(probabilities), axis=-1)

    return entropy


def compute_probability_margin(probabilities: np.ndarray) -> np.ndarray:
    """
    Compute margin between top-1 and top-2 predictions.

    Large margin = confident prediction.

    Args:
        probabilities: Probability distributions (shape: [n_samples, n_classes])

    Returns:
        Margin values (shape: [n_samples])
    """
    # Sort probabilities in descending order
    sorted_probs = np.sort(probabilities, axis=-1)[:, ::-1]

    # Margin = P(top1) - P(top2)
    if sorted_probs.shape[1] >= 2:
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    else:
        # Only one class
        margin = sorted_probs[:, 0]

    return margin


def compute_top_k_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    k: int = 3,
) -> float:
    """
    Compute top-k accuracy.

    Args:
        predictions: Top-k predictions (shape: [n_samples, k])
        labels: Ground truth labels (shape: [n_samples])
        k: Number of top predictions

    Returns:
        Top-k accuracy
    """
    if predictions.ndim == 1:
        # Single predictions - treat as top-1
        return np.mean(predictions == labels)

    # Check if label is in top-k predictions
    correct = np.any(predictions[:, :k] == labels[:, np.newaxis], axis=1)
    top_k_acc = np.mean(correct)

    return float(top_k_acc)
