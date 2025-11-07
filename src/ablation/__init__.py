"""Ablation experiment framework for attention head analysis."""

from .combination import run_combination_analysis
from .evaluator import AblationEvaluator
from .head_ablation import run_head_ablation
from .hooks import AblationHookManager, HeadAblationHook
from .layer_ablation import run_layer_ablation
from .methods import mean_ablation, random_ablation, zero_ablation
from .statistics import (
    bootstrap_confidence_interval,
    compute_effect_size,
    multiple_comparison_correction,
    permutation_test,
)

__all__ = [
    "AblationEvaluator",
    "AblationHookManager",
    "HeadAblationHook",
    "bootstrap_confidence_interval",
    "compute_effect_size",
    "mean_ablation",
    "multiple_comparison_correction",
    "permutation_test",
    "random_ablation",
    "run_combination_analysis",
    "run_head_ablation",
    "run_layer_ablation",
    "zero_ablation",
]
