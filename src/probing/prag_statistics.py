"""Statistical validation for PRAG experiments."""

from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import pearsonr, wilcoxon


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Args:
        x: First sample
        y: Second sample

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(x), len(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    # Cohen's d
    d = (np.mean(x) - np.mean(y)) / pooled_std
    return float(d)


class PRAGStatistics:
    """Statistical validation for PRAG experiments."""

    def __init__(self, n_bootstrap: int = 1000, random_seed: int = 42) -> None:
        """
        Initialize PRAG statistics.

        Args:
            n_bootstrap: Number of bootstrap samples
            random_seed: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def test_vlm_vs_llm(
        self,
        vlm_prag: np.ndarray | list[float],
        llm_prag: np.ndarray | list[float],
    ) -> dict[str, Any]:
        """
        Test if VLM PRAG is significantly lower than LLM PRAG.

        Uses Wilcoxon signed-rank test and Cohen's d.

        Args:
            vlm_prag: PRAG scores for VLM condition
            llm_prag: PRAG scores for LLM condition

        Returns:
            Dictionary containing:
                - statistic: Wilcoxon test statistic
                - p_value: P-value
                - effect_size: Cohen's d
                - vlm_mean: Mean PRAG for VLM
                - llm_mean: Mean PRAG for LLM
                - gap: LLM_mean - VLM_mean
                - significant: Whether difference is significant (p < 0.05 and |d| > 0.5)
        """
        vlm_prag = np.array(vlm_prag)
        llm_prag = np.array(llm_prag)

        if len(vlm_prag) != len(llm_prag):
            error_msg = f"VLM and LLM PRAG arrays must have same length: {len(vlm_prag)} vs {len(llm_prag)}"
            raise ValueError(error_msg)

        # Wilcoxon signed-rank test
        statistic, p_value = wilcoxon(vlm_prag, llm_prag, alternative="less")

        # Cohen's d
        effect_size = cohen_d(llm_prag, vlm_prag)  # Positive if LLM > VLM

        # Means
        vlm_mean = float(np.mean(vlm_prag))
        llm_mean = float(np.mean(llm_prag))
        gap = llm_mean - vlm_mean

        # Significance: p < 0.05 and effect size > 0.5
        significant = p_value < 0.05 and abs(effect_size) > 0.5

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "vlm_mean": vlm_mean,
            "llm_mean": llm_mean,
            "gap": float(gap),
            "significant": significant,
            "n": len(vlm_prag),
        }

    def test_prag_predicts_performance(
        self,
        prag_scores: np.ndarray | list[float],
        performance_scores: np.ndarray | list[float],
    ) -> dict[str, Any]:
        """
        Test if PRAG predicts performance (decode accuracy or performance gap).

        Uses Pearson correlation with bootstrap confidence intervals.

        Args:
            prag_scores: PRAG scores
            performance_scores: Performance scores (e.g., decode accuracy)

        Returns:
            Dictionary containing:
                - correlation: Pearson correlation coefficient
                - p_value: P-value for correlation test
                - ci_95: 95% bootstrap confidence interval
                - ci_lower: Lower bound of CI
                - ci_upper: Upper bound of CI
                - significant: Whether correlation is significant (p < 0.05)
        """
        prag_scores = np.array(prag_scores)
        performance_scores = np.array(performance_scores)

        if len(prag_scores) != len(performance_scores):
            error_msg = f"PRAG and performance arrays must have same length: {len(prag_scores)} vs {len(performance_scores)}"
            raise ValueError(error_msg)

        # Pearson correlation
        r, p_value = pearsonr(prag_scores, performance_scores)

        # Bootstrap confidence interval
        r_bootstrap = []
        n = len(prag_scores)
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            r_boot, _ = pearsonr(prag_scores[idx], performance_scores[idx])
            r_bootstrap.append(r_boot)

        ci_lower, ci_upper = np.percentile(r_bootstrap, [2.5, 97.5])

        significant = p_value < 0.05

        return {
            "correlation": float(r),
            "p_value": float(p_value),
            "ci_95": (float(ci_lower), float(ci_upper)),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "significant": significant,
            "n": n,
        }

    def test_attribute_differences(
        self,
        continuous_prag: np.ndarray | list[float],
        discrete_prag: np.ndarray | list[float],
    ) -> dict[str, Any]:
        """
        Test if continuous attributes have lower PRAG than discrete attributes.

        Uses Mann-Whitney U test (non-parametric) and Cohen's d.

        Args:
            continuous_prag: PRAG scores for continuous attributes
            discrete_prag: PRAG scores for discrete attributes

        Returns:
            Dictionary containing test results
        """
        continuous_prag = np.array(continuous_prag)
        discrete_prag = np.array(discrete_prag)

        # Mann-Whitney U test (two-sided)
        statistic, p_value = stats.mannwhitneyu(continuous_prag, discrete_prag, alternative="less")

        # Cohen's d
        effect_size = cohen_d(discrete_prag, continuous_prag)  # Positive if discrete > continuous

        # Means
        continuous_mean = float(np.mean(continuous_prag))
        discrete_mean = float(np.mean(discrete_prag))
        gap = discrete_mean - continuous_mean

        significant = p_value < 0.05 and abs(effect_size) > 0.5

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "continuous_mean": continuous_mean,
            "discrete_mean": discrete_mean,
            "gap": float(gap),
            "significant": significant,
            "n_continuous": len(continuous_prag),
            "n_discrete": len(discrete_prag),
        }
