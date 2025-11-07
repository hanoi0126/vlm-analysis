"""Head-level ablation experiment.

This experiment identifies specific attention heads responsible for
vision-language alignment by ablating individual heads and measuring
performance impact.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.utils.paths import get_model_short_name

from .evaluator import AblationEvaluator
from .statistics import multiple_comparison_correction


def run_head_ablation(
    model: Any,
    processor: Any,
    config: Any,
    target_layers: list[int] | None = None,
    tasks: list[str] | None = None,
    output_dir: str | Path | None = None,
    device: str = "cuda",
    num_heads: int = 28,
    show_progress: bool = True,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    model_id: str | None = None,
) -> pd.DataFrame:
    """
    Run head-level ablation experiment.

    This is the core experiment that identifies alignment heads by ablating
    individual attention heads in critical layers.

    Args:
        model: VLM model
        processor: Model processor
        config: Experiment configuration
        target_layers: List of layers to analyze (default: load from layer ablation results)
        tasks: List of tasks to evaluate (default: use config.experiment.tasks)
        output_dir: Output directory (default: config.output.results_root/ablation/head/{model_id})
        device: Device to run on
        num_heads: Number of attention heads per layer
        show_progress: Show progress bars
        n_bootstrap: Number of bootstrap samples
        n_permutations: Number of permutations for significance test
        model_id: Model identifier for organizing results (default: from config)

    Returns:
        DataFrame with head ablation results
    """
    # Setup
    if tasks is None:
        tasks = config.experiment.tasks

    # Get model short name for directory structure
    if model_id is None:
        model_id = get_model_short_name(config.model.model_id)  # e.g., qwen25_3b

    if output_dir is None:
        output_dir = Path(config.output.results_root) / "ablation" / "head" / model_id
    else:
        output_dir = Path(output_dir)

    # Create subdirectories
    by_layer_dir = output_dir / "by_layer"
    by_task_dir = output_dir / "by_task"
    summary_dir = output_dir / "summary"
    plots_dir = output_dir / "plots"

    for dir_path in [by_layer_dir, by_task_dir, summary_dir, plots_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("\nOutput directory structure:")
    print(f"  Base: {output_dir}")
    print(f"  By layer: {by_layer_dir}")
    print(f"  By task: {by_task_dir}")
    print(f"  Summary: {summary_dir}")
    print(f"  Plots: {plots_dir}")

    # Load target layers if not provided
    if target_layers is None:
        layer_dir = Path(config.output.results_root) / "ablation" / "layer"
        critical_layers_path = layer_dir / "critical_layers.json"

        if critical_layers_path.exists():
            with open(critical_layers_path) as f:
                critical_info = json.load(f)
                target_layers = critical_info["critical_layers"]
            print(f"Loaded critical layers from layer ablation: {target_layers}")
        else:
            # Default: analyze middle-to-late layers
            target_layers = [14, 15, 16, 17]
            print(f"Using default target layers: {target_layers}")

    # Initialize evaluator
    evaluator = AblationEvaluator(
        model=model,
        processor=processor,
        device=device,
        batch_size=config.batch_size,
        num_heads=num_heads,
        evaluation_position="answer_head",
    )

    # Get dataset name
    dataset_name = config.dataset.hf_dataset
    split = config.dataset.hf_split

    # Compute baseline for each task
    print("=" * 80)
    print("Computing baseline (no ablation)...")
    print("=" * 80)

    baseline_results = {}
    for task in tasks:
        print(f"\n[Baseline] Task: {task}")
        baseline = evaluator.evaluate_task(
            dataset_name=dataset_name,
            task=task,
            split=split,
            layer_idx=None,
            head_idx=None,
            show_progress=show_progress,
        )
        baseline_results[task] = baseline

        print(f"  Accuracy: {baseline['accuracy']:.4f}")
        print(f"  Samples: {baseline['n_samples']}")

    # Save baseline
    baseline_df = pd.DataFrame(
        [
            {
                "layer_idx": -1,
                "head_idx": -1,
                "task": task,
                "n_samples": results["n_samples"],
                "baseline_acc": results["accuracy"],
                "ablated_acc": results["accuracy"],
                "delta_acc": 0.0,
            }
            for task, results in baseline_results.items()
        ]
    )
    baseline_df.to_csv(summary_dir / "baseline.csv", index=False)

    # Head-by-head ablation
    print("\n" + "=" * 80)
    print("Running head-level ablation...")
    print(f"Target layers: {target_layers}")
    print(f"Heads per layer: {num_heads}")
    print(f"Total combinations: {len(target_layers)} layers x {num_heads} heads x {len(tasks)} tasks")
    print("=" * 80)

    results = []
    total_iterations = len(target_layers) * num_heads * len(tasks)

    with tqdm(total=total_iterations, desc="Head ablation") as pbar:
        for layer_idx in target_layers:
            for head_idx in range(num_heads):
                for task in tasks:
                    pbar.set_description(f"L{layer_idx} H{head_idx:02d} {task}")

                    # Ablate single head
                    ablated = evaluator.evaluate_task(
                        dataset_name=dataset_name,
                        task=task,
                        split=split,
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        show_progress=False,
                    )

                    # Compare with baseline
                    comparison = evaluator.compare_with_baseline(
                        baseline_results[task],
                        ablated,
                        compute_statistics=True,
                        n_bootstrap=n_bootstrap,
                        n_permutations=n_permutations,
                    )

                    # Record results
                    result = {
                        "layer_idx": layer_idx,
                        "head_idx": head_idx,
                        "task": task,
                        "n_samples": comparison["n_samples"],
                        "baseline_acc": comparison["baseline_acc"],
                        "ablated_acc": comparison["ablated_acc"],
                        "ablated_ci_lower": comparison.get("ablated_ci_lower", np.nan),
                        "ablated_ci_upper": comparison.get("ablated_ci_upper", np.nan),
                        "delta_acc": comparison["delta_acc"],
                        "p_value": comparison.get("p_value", np.nan),
                        "effect_size": comparison.get("effect_size", np.nan),
                        "is_significant": comparison.get("is_significant", False),
                        "baseline_top3": comparison["baseline_top3"],
                        "ablated_top3": comparison["ablated_top3"],
                        "delta_top3": comparison["delta_top3"],
                    }
                    results.append(result)

                    pbar.update(1)

                    # Log significant heads immediately
                    if comparison.get("is_significant", False):
                        tqdm.write(f"  ⚠️  SIGNIFICANT: L{layer_idx} H{head_idx:02d} {task}: Δacc={comparison['delta_acc']:+.4f} p<0.001")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Add rank within each layer-task combination
    df["rank_in_layer"] = df.groupby(["layer_idx", "task"])["delta_acc"].rank(method="dense", ascending=True)

    # Save detailed results to summary/
    detailed_path = summary_dir / "head_ablation_detailed.csv"
    df.to_csv(detailed_path, index=False)
    print(f"\nSaved detailed results to: {detailed_path}")

    # Multiple comparison correction
    print("\nApplying multiple comparison correction...")
    df = apply_multiple_comparison_correction(df)

    # Save corrected results to summary/
    corrected_path = summary_dir / "head_ablation_corrected.csv"
    df.to_csv(corrected_path, index=False)
    print(f"Saved corrected results to: {corrected_path}")

    # Save split results by layer and task
    print("\nSaving results by layer and task...")
    save_results_by_layer(df, by_layer_dir, target_layers)
    save_results_by_task(df, by_task_dir, tasks)

    # Identify alignment heads
    alignment_heads = identify_alignment_heads(
        df,
        delta_threshold=-0.30,
        p_threshold=0.001,
        use_corrected_p=True,
    )

    # Save alignment heads summary to summary/
    alignment_path = summary_dir / "alignment_heads_summary.json"
    with open(alignment_path, "w") as f:
        json.dump(alignment_heads, f, indent=2)
    print(f"\nSaved alignment heads to: {alignment_path}")

    # Create task specificity matrix
    task_matrix = create_task_specificity_matrix(df, target_layers, num_heads, tasks)
    matrix_path = summary_dir / "task_specificity_matrix.csv"
    task_matrix.to_csv(matrix_path)
    print(f"Saved task specificity matrix to: {matrix_path}")

    # Print summary
    print_head_ablation_summary(alignment_heads)

    return df


def save_results_by_layer(df: pd.DataFrame, by_layer_dir: Path, target_layers: list[int]) -> None:
    """
    Save results split by layer.

    Args:
        df: DataFrame with all results
        by_layer_dir: Directory to save layer-specific files
        target_layers: List of target layers
    """
    for layer_idx in target_layers:
        layer_df = df[df["layer_idx"] == layer_idx].copy()
        if len(layer_df) > 0:
            layer_file = by_layer_dir / f"layer_{layer_idx:02d}.csv"
            layer_df.to_csv(layer_file, index=False)
            print(f"  Saved layer {layer_idx} results: {layer_file.name}")


def save_results_by_task(df: pd.DataFrame, by_task_dir: Path, tasks: list[str]) -> None:
    """
    Save results split by task.

    Args:
        df: DataFrame with all results
        by_task_dir: Directory to save task-specific files
        tasks: List of tasks
    """
    for task in tasks:
        task_df = df[df["task"] == task].copy()
        if len(task_df) > 0:
            task_file = by_task_dir / f"{task}.csv"
            task_df.to_csv(task_file, index=False)
            print(f"  Saved task '{task}' results: {task_file.name}")


def apply_multiple_comparison_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply multiple comparison correction to p-values.

    Args:
        df: DataFrame with uncorrected p-values

    Returns:
        DataFrame with corrected p-values and updated significance flags
    """
    p_values = df["p_value"].to_numpy()

    # Apply Bonferroni correction
    corrected_p, is_significant_corrected = multiple_comparison_correction(
        p_values,
        method="bonferroni",
        alpha=0.001,
    )

    df["p_value_corrected"] = corrected_p
    df["is_significant_corrected"] = is_significant_corrected

    # Also apply FDR for comparison
    fdr_p, is_significant_fdr = multiple_comparison_correction(
        p_values,
        method="fdr",
        alpha=0.05,
    )

    df["p_value_fdr"] = fdr_p
    df["is_significant_fdr"] = is_significant_fdr

    return df


def identify_alignment_heads(
    df: pd.DataFrame,
    delta_threshold: float = -0.30,
    p_threshold: float = 0.001,
    use_corrected_p: bool = True,
) -> dict[str, Any]:
    """
    Identify alignment heads based on performance degradation.

    Args:
        df: DataFrame with head ablation results
        delta_threshold: Delta accuracy threshold (negative)
        p_threshold: P-value threshold
        use_corrected_p: Use corrected p-values

    Returns:
        Dictionary with alignment head information
    """
    p_col = "p_value_corrected" if use_corrected_p and "p_value_corrected" in df.columns else "p_value"

    # Filter significant heads
    significant_df = df[(df["delta_acc"] < delta_threshold) & (df[p_col] < p_threshold)].copy()

    if len(significant_df) == 0:
        return {
            "alignment_heads": [],
            "summary": {
                "total_important_heads": 0,
                "task_agnostic_heads": 0,
                "task_specific_heads": 0,
            },
        }

    # Group by layer and head
    head_groups = significant_df.groupby(["layer_idx", "head_idx"])

    alignment_heads_list = []
    for (layer_idx, head_idx), group in head_groups:
        tasks_affected = group["task"].tolist()
        avg_delta = group["delta_acc"].mean()
        min_p = group[p_col].min()

        # Determine if task-agnostic or task-specific
        head_type = "task_agnostic" if len(tasks_affected) >= 3 else "task_specific"

        head_info = {
            "layer": int(layer_idx),
            "head": int(head_idx),
            "tasks_affected": tasks_affected,
            "avg_delta_acc": float(avg_delta),
            "min_p_value": f"<{p_threshold}" if min_p < p_threshold else f"{min_p:.2e}",
            "type": head_type,
            "description": _generate_head_description(layer_idx, head_idx, tasks_affected, head_type),
        }
        alignment_heads_list.append(head_info)

    # Sort by average delta (most negative first)
    alignment_heads_list.sort(key=lambda x: x["avg_delta_acc"])

    # Count types
    task_agnostic_count = sum(1 for h in alignment_heads_list if h["type"] == "task_agnostic")
    task_specific_count = sum(1 for h in alignment_heads_list if h["type"] == "task_specific")

    # Find most critical layer
    layer_counts = significant_df.groupby("layer_idx").size()
    most_critical_layer = int(layer_counts.idxmax()) if len(layer_counts) > 0 else None

    # Compute average effect size
    avg_effect_size = significant_df["effect_size"].mean() if "effect_size" in significant_df.columns else np.nan

    summary = {
        "total_important_heads": len(alignment_heads_list),
        "task_agnostic_heads": task_agnostic_count,
        "task_specific_heads": task_specific_count,
        "most_critical_layer": most_critical_layer,
        "avg_effect_size": float(avg_effect_size) if not np.isnan(avg_effect_size) else None,
        "thresholds": {
            "delta_acc": delta_threshold,
            "p_value": p_threshold,
            "correction_method": "bonferroni" if use_corrected_p else "none",
        },
    }

    return {
        "alignment_heads": alignment_heads_list,
        "summary": summary,
    }


def _generate_head_description(
    _layer_idx: int,
    _head_idx: int,
    tasks_affected: list[str],
    head_type: str,
) -> str:
    """Generate human-readable description for an alignment head."""
    if head_type == "task_agnostic":
        return f"General vision-to-language alignment head (affects {len(tasks_affected)} tasks)"
    task_str = ", ".join(tasks_affected)
    return f"Specialized head for {task_str}"


def create_task_specificity_matrix(
    df: pd.DataFrame,
    layers: list[int],
    num_heads: int,
    tasks: list[str],
) -> pd.DataFrame:
    """
    Create a matrix showing head importance for each task.

    Args:
        df: DataFrame with head ablation results
        layers: List of layer indices
        num_heads: Number of heads per layer
        tasks: List of tasks

    Returns:
        DataFrame with head × task importance matrix
    """
    # Create pivot table: rows = (layer, head), columns = tasks, values = delta_acc
    matrix_data = []

    for layer_idx in layers:
        for head_idx in range(num_heads):
            row = {
                "layer_idx": layer_idx,
                "head_idx": head_idx,
                "head_id": f"L{layer_idx}_H{head_idx:02d}",
            }

            for task in tasks:
                # Find delta_acc for this combination
                subset = df[(df["layer_idx"] == layer_idx) & (df["head_idx"] == head_idx) & (df["task"] == task)]

                if len(subset) > 0:
                    row[f"delta_{task}"] = subset.iloc[0]["delta_acc"]
                else:
                    row[f"delta_{task}"] = np.nan

            matrix_data.append(row)

    matrix_df = pd.DataFrame(matrix_data)
    return matrix_df


def print_head_ablation_summary(alignment_heads: dict[str, Any]) -> None:
    """Print a summary of head ablation results."""
    print("\n" + "=" * 80)
    print("Head Ablation Summary: Alignment Heads")
    print("=" * 80)

    summary = alignment_heads["summary"]
    print(f"\nTotal important heads: {summary['total_important_heads']}")
    print(f"  - Task-agnostic heads: {summary['task_agnostic_heads']}")
    print(f"  - Task-specific heads: {summary['task_specific_heads']}")

    if summary.get("most_critical_layer") is not None:
        print(f"\nMost critical layer: {summary['most_critical_layer']}")

    if summary.get("avg_effect_size") is not None:
        print(f"Average effect size: {summary['avg_effect_size']:.2f}")

    print("\n" + "-" * 80)
    print("Top Alignment Heads:")
    print("-" * 80)

    for i, head in enumerate(alignment_heads["alignment_heads"][:10], 1):
        print(f"\n{i}. Layer {head['layer']}, Head {head['head']} ({head['type']})")
        print(f"   Avg Δacc: {head['avg_delta_acc']:+.4f}")
        print(f"   Tasks: {', '.join(head['tasks_affected'])}")
        print(f"   P-value: {head['min_p_value']}")
        print(f"   {head['description']}")

    print("\n" + "=" * 80)
