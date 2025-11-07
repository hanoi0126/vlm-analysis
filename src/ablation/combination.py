"""Multi-head combination analysis.

This experiment analyzes how multiple alignment heads interact,
testing for redundancy and complementarity by ablating combinations
of heads simultaneously.
"""

from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.ablation import AblationEvaluator, AblationHookManager
from src.data import HuggingFaceDataset


def run_combination_analysis(
    model: Any,
    processor: Any,
    config: Any,
    alignment_heads: list[tuple[int, int]] | None = None,
    tasks: list[str] | None = None,
    output_dir: str | Path | None = None,
    device: str = "cuda",
    max_combinations: int = 3,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Run multi-head combination ablation analysis.

    Tests redundancy and complementarity by ablating multiple heads simultaneously.

    Args:
        model: VLM model
        processor: Model processor
        config: Experiment configuration
        alignment_heads: List of (layer_idx, head_idx) tuples (load from Phase 2 if None)
        tasks: List of tasks to evaluate
        output_dir: Output directory
        device: Device to run on
        max_combinations: Maximum number of heads to ablate simultaneously
        show_progress: Show progress bars

    Returns:
        DataFrame with combination analysis results
    """
    # Setup
    if tasks is None:
        tasks = config.experiment.tasks

    if output_dir is None:
        output_dir = Path(config.output.results_root) / "ablation" / "phase3"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load alignment heads if not provided
    if alignment_heads is None:
        phase2_dir = Path(config.output.results_root) / "ablation" / "phase2"
        alignment_path = phase2_dir / "alignment_heads_summary.json"

        if alignment_path.exists():
            with open(alignment_path) as f:
                alignment_info = json.load(f)
                alignment_heads = [
                    (h["layer"], h["head"])
                    for h in alignment_info["alignment_heads"][:15]  # Top 15 heads
                ]
            print(f"Loaded {len(alignment_heads)} alignment heads from Phase 2")
        else:
            msg = "No alignment heads found. Run Phase 2 first."
            raise ValueError(msg)

    # Initialize evaluator
    evaluator = AblationEvaluator(
        model=model,
        processor=processor,
        device=device,
        batch_size=config.batch_size,
        num_heads=28,
        evaluation_position="answer_head",
    )

    # Get dataset name
    dataset_name = config.dataset.hf_dataset
    split = config.dataset.hf_split

    # Compute baseline
    print("=" * 80)
    print("Computing baseline...")
    print("=" * 80)

    baseline_results = {}
    for task in tasks:
        baseline = evaluator.evaluate_task(
            dataset_name=dataset_name,
            task=task,
            split=split,
            layer_idx=None,
            head_idx=None,
            show_progress=show_progress,
        )
        baseline_results[task] = baseline

    # Load single-head ablation results for comparison
    phase2_df = None
    phase2_path = Path(config.output.results_root) / "ablation" / "phase2" / "head_ablation_detailed.csv"
    if phase2_path.exists():
        phase2_df = pd.read_csv(phase2_path)
        print("Loaded Phase 2 results for comparison")

    # Run combination experiments
    print("\n" + "=" * 80)
    print("Running multi-head combination analysis...")
    print(f"Alignment heads: {len(alignment_heads)}")
    print(f"Max combination size: {max_combinations}")
    print("=" * 80)

    results = []

    # Test pairwise combinations
    for combo_size in range(1, min(max_combinations, len(alignment_heads)) + 1):
        print(f"\n--- Testing {combo_size}-head combinations ---")

        combos = list(combinations(alignment_heads, combo_size))
        print(f"Total combinations: {len(combos)}")

        for combo in tqdm(combos, desc=f"{combo_size}-head combos", disable=not show_progress):
            for task in tasks:
                # Ablate multiple heads simultaneously
                ablated = evaluate_with_multi_head_ablation(
                    evaluator=evaluator,
                    model=model,
                    dataset_name=dataset_name,
                    task=task,
                    split=split,
                    heads_to_ablate=combo,
                )

                # Compare with baseline
                comparison = evaluator.compare_with_baseline(
                    baseline_results[task],
                    ablated,
                    compute_statistics=True,
                    n_bootstrap=100,
                    n_permutations=100,
                )

                # Get expected effect from individual ablations
                expected_effect = compute_expected_effect(combo, task, phase2_df, baseline_results[task]["accuracy"])

                # Record results
                result = {
                    "combo_size": combo_size,
                    "heads": str(combo),
                    "task": task,
                    "n_samples": comparison["n_samples"],
                    "baseline_acc": comparison["baseline_acc"],
                    "ablated_acc": comparison["ablated_acc"],
                    "delta_acc": comparison["delta_acc"],
                    "expected_delta": expected_effect,
                    "interaction_effect": comparison["delta_acc"] - expected_effect,
                    "p_value": comparison.get("p_value", np.nan),
                    "is_significant": comparison.get("is_significant", False),
                }
                results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save results
    detailed_path = output_dir / "combination_analysis.csv"
    df.to_csv(detailed_path, index=False)
    print(f"\nSaved results to: {detailed_path}")

    # Analyze redundancy and complementarity
    analysis = analyze_head_interactions(df, alignment_heads)

    # Save analysis
    analysis_path = output_dir / "interaction_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved interaction analysis to: {analysis_path}")

    # Print summary
    print_phase3_summary(analysis)

    return df


def evaluate_with_multi_head_ablation(
    evaluator: AblationEvaluator,
    model: Any,
    dataset_name: str,
    task: str,
    split: str,
    heads_to_ablate: list[tuple[int, int]],
) -> dict[str, Any]:
    """
    Evaluate model with multiple heads ablated simultaneously.

    Args:
        evaluator: AblationEvaluator instance
        model: The model
        dataset_name: Dataset name
        task: Task name
        split: Dataset split
        heads_to_ablate: List of (layer_idx, head_idx) tuples

    Returns:
        Evaluation results
    """
    # Register multiple hooks
    hook_manager = AblationHookManager()

    for layer_idx, head_idx in heads_to_ablate:
        hook_manager.register_hook(
            model=model,
            layer_idx=layer_idx,
            head_idx=head_idx,
            ablation_type="zero",
            num_heads=28,
        )

    try:
        # Load dataset
        dataset = HuggingFaceDataset(
            dataset_name=dataset_name,
            split=split if split != "auto" else task,
            task=None if split == task else task,
        )

        # Evaluate with hooks active
        results = evaluator.evaluate_with_ablation(
            dataset=dataset,
            layer_idx=None,  # Hooks already registered
            head_idx=None,
            show_progress=False,
        )

    finally:
        # Remove all hooks
        hook_manager.remove_all_hooks()

    return results


def compute_expected_effect(
    combo: tuple[tuple[int, int], ...],
    task: str,
    phase2_df: pd.DataFrame | None,
    baseline_acc: float,
) -> float:
    """
    Compute expected effect assuming independence.

    If heads are independent, the combined effect should be approximately
    the sum of individual effects.

    Args:
        combo: Tuple of (layer_idx, head_idx) tuples
        task: Task name
        phase2_df: Phase 2 results DataFrame
        baseline_acc: Baseline accuracy

    Returns:
        Expected delta accuracy
    """
    if phase2_df is None:
        return np.nan

    individual_deltas = []
    for layer_idx, head_idx in combo:
        subset = phase2_df[(phase2_df["layer_idx"] == layer_idx) & (phase2_df["head_idx"] == head_idx) & (phase2_df["task"] == task)]

        if len(subset) > 0:
            individual_deltas.append(subset.iloc[0]["delta_acc"])

    if not individual_deltas:
        return np.nan

    # Simple additive model (alternative: multiplicative on accuracy)
    expected_delta = sum(individual_deltas)

    # Clip to valid range
    expected_delta = max(expected_delta, -baseline_acc)

    return expected_delta


def analyze_head_interactions(
    df: pd.DataFrame,
    _alignment_heads: list[tuple[int, int]],
) -> dict[str, Any]:
    """
    Analyze redundancy and complementarity in head interactions.

    Redundancy: Combined effect < sum of individual effects (heads overlap)
    Complementarity: Combined effect > sum of individual effects (heads synergize)

    Args:
        df: DataFrame with combination results
        alignment_heads: List of alignment heads

    Returns:
        Dictionary with interaction analysis
    """
    # Focus on pairwise combinations
    pairwise_df = df[df["combo_size"] == 2].copy()

    if len(pairwise_df) == 0:
        return {
            "redundant_pairs": [],
            "complementary_pairs": [],
            "independent_pairs": [],
            "summary": {
                "total_pairs": 0,
                "redundant_count": 0,
                "complementary_count": 0,
                "independent_count": 0,
            },
        }

    # Classify interactions
    pairwise_df["interaction_type"] = "independent"
    pairwise_df.loc[pairwise_df["interaction_effect"] < -0.05, "interaction_type"] = "redundant"
    pairwise_df.loc[pairwise_df["interaction_effect"] > 0.05, "interaction_type"] = "complementary"

    # Find interesting pairs
    redundant_pairs = []
    complementary_pairs = []
    independent_pairs = []

    for interaction_type in ["redundant", "complementary", "independent"]:
        subset = pairwise_df[pairwise_df["interaction_type"] == interaction_type]

        for _, row in subset.iterrows():
            pair_info = {
                "heads": row["heads"],
                "task": row["task"],
                "observed_delta": float(row["delta_acc"]),
                "expected_delta": float(row["expected_delta"]) if not np.isnan(row["expected_delta"]) else None,
                "interaction_effect": float(row["interaction_effect"]) if not np.isnan(row["interaction_effect"]) else None,
            }

            if interaction_type == "redundant":
                redundant_pairs.append(pair_info)
            elif interaction_type == "complementary":
                complementary_pairs.append(pair_info)
            else:
                independent_pairs.append(pair_info)

    # Sort by interaction strength
    redundant_pairs.sort(key=lambda x: x["interaction_effect"] if x["interaction_effect"] else 0)
    complementary_pairs.sort(key=lambda x: -(x["interaction_effect"] if x["interaction_effect"] else 0))

    summary = {
        "total_pairs": len(pairwise_df),
        "redundant_count": len(redundant_pairs),
        "complementary_count": len(complementary_pairs),
        "independent_count": len(independent_pairs),
        "avg_interaction_effect": float(pairwise_df["interaction_effect"].mean()) if len(pairwise_df) > 0 else None,
    }

    return {
        "redundant_pairs": redundant_pairs[:20],  # Top 20
        "complementary_pairs": complementary_pairs[:20],
        "independent_pairs": independent_pairs[:10],
        "summary": summary,
    }


def print_phase3_summary(analysis: dict[str, Any]) -> None:
    """Print a summary of Phase 3 results."""
    print("\n" + "=" * 80)
    print("Phase 3 Summary: Head Interactions")
    print("=" * 80)

    summary = analysis["summary"]
    print(f"\nTotal pairs analyzed: {summary['total_pairs']}")
    print(f"  - Redundant pairs: {summary['redundant_count']}")
    print(f"  - Complementary pairs: {summary['complementary_count']}")
    print(f"  - Independent pairs: {summary['independent_count']}")

    if summary.get("avg_interaction_effect") is not None:
        print(f"\nAverage interaction effect: {summary['avg_interaction_effect']:+.4f}")

    # Show top redundant pairs
    if analysis["redundant_pairs"]:
        print("\n" + "-" * 80)
        print("Top Redundant Pairs (overlapping function):")
        print("-" * 80)

        for i, pair in enumerate(analysis["redundant_pairs"][:5], 1):
            print(f"\n{i}. {pair['heads']} - {pair['task']}")
            print(f"   Observed: {pair['observed_delta']:+.4f}")
            print(f"   Expected: {pair['expected_delta']:+.4f}")
            print(f"   Redundancy: {pair['interaction_effect']:+.4f}")

    # Show top complementary pairs
    if analysis["complementary_pairs"]:
        print("\n" + "-" * 80)
        print("Top Complementary Pairs (synergistic):")
        print("-" * 80)

        for i, pair in enumerate(analysis["complementary_pairs"][:5], 1):
            print(f"\n{i}. {pair['heads']} - {pair['task']}")
            print(f"   Observed: {pair['observed_delta']:+.4f}")
            print(f"   Expected: {pair['expected_delta']:+.4f}")
            print(f"   Synergy: {pair['interaction_effect']:+.4f}")

    print("\n" + "=" * 80)
