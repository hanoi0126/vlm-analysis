"""Layer-level ablation experiment.

This script identifies critical layers by ablating all attention heads
in each layer and measuring performance degradation.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.ablation import AblationEvaluator


def run_layer_ablation(
    model: Any,
    processor: Any,
    config: Any,
    tasks: list[str] | None = None,
    output_dir: str | Path | None = None,
    device: str = "cuda",
    num_layers: int = 28,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Run layer-level ablation experiment.

    Ablates all attention heads in each layer and measures performance
    degradation across tasks.

    Args:
        model: VLM model
        processor: Model processor
        config: Experiment configuration
        tasks: List of tasks to evaluate (default: use config.experiment.tasks)
        output_dir: Output directory (default: config.output.results_root/ablation/phase1)
        device: Device to run on
        num_layers: Number of layers in the model
        show_progress: Show progress bars

    Returns:
        DataFrame with layer ablation results
    """
    # Setup
    if tasks is None:
        tasks = config.experiment.tasks

    if output_dir is None:
        output_dir = Path(config.output.results_root) / "ablation" / "phase1"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize evaluator
    evaluator = AblationEvaluator(
        model=model,
        processor=processor,
        device=device,
        batch_size=config.batch_size,
        num_heads=28,  # Qwen2.5-VL has 28 query heads
        evaluation_position="answer_head",
    )

    # Get dataset name
    dataset_name = config.dataset.hf_dataset
    split = config.dataset.hf_split

    results = []

    # Compute baseline for each task (no ablation)
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
            layer_idx=None,  # No ablation
            head_idx=None,
            show_progress=show_progress,
        )
        baseline_results[task] = baseline

        print(f"  Accuracy: {baseline['accuracy']:.4f}")
        print(f"  Top-3 Accuracy: {baseline['top3_accuracy']:.4f}")
        print(f"  Samples: {baseline['n_samples']}")

    # Save baseline results
    baseline_df = pd.DataFrame(
        [
            {
                "layer_idx": -1,  # -1 indicates baseline
                "task": task,
                "n_samples": results["n_samples"],
                "baseline_acc": results["accuracy"],
                "ablated_acc": results["accuracy"],
                "delta_acc": 0.0,
                "top3_acc": results["top3_accuracy"],
            }
            for task, results in baseline_results.items()
        ]
    )
    baseline_df.to_csv(output_dir / "baseline.csv", index=False)
    print(f"\nSaved baseline to: {output_dir / 'baseline.csv'}")

    # Layer-by-layer ablation
    print("\n" + "=" * 80)
    print("Running layer-level ablation...")
    print("=" * 80)

    for layer_idx in tqdm(range(num_layers), desc="Layers", disable=not show_progress):
        print(f"\n--- Layer {layer_idx} ---")

        for task in tasks:
            print(f"  Task: {task}")

            # Ablate entire layer (all heads)
            ablated = evaluator.evaluate_task(
                dataset_name=dataset_name,
                task=task,
                split=split,
                layer_idx=layer_idx,
                head_idx=None,  # None = ablate all heads in layer
                show_progress=False,  # Suppress nested progress bar
            )

            # Compare with baseline
            comparison = evaluator.compare_with_baseline(
                baseline_results[task],
                ablated,
                compute_statistics=True,
                n_bootstrap=100,  # Use fewer samples for speed in Phase 1
                n_permutations=100,
            )

            # Record results
            result = {
                "layer_idx": layer_idx,
                "task": task,
                "n_samples": comparison["n_samples"],
                "baseline_acc": comparison["baseline_acc"],
                "ablated_acc": comparison["ablated_acc"],
                "delta_acc": comparison["delta_acc"],
                "ablated_ci_lower": comparison.get("ablated_ci_lower", np.nan),
                "ablated_ci_upper": comparison.get("ablated_ci_upper", np.nan),
                "p_value": comparison.get("p_value", np.nan),
                "effect_size": comparison.get("effect_size", np.nan),
                "is_significant": comparison.get("is_significant", False),
                "top3_acc": comparison["ablated_top3"],
                "delta_top3": comparison["delta_top3"],
            }
            results.append(result)

            print(f"    Baseline: {comparison['baseline_acc']:.4f}")
            print(f"    Ablated:  {comparison['ablated_acc']:.4f}")
            print(f"    Delta:    {comparison['delta_acc']:+.4f}")
            if "p_value" in comparison:
                p_str = f"<{1e-3:.1e}" if comparison["p_value"] < 1e-3 else f"{comparison['p_value']:.4f}"
                print(f"    P-value:  {p_str}")
                print(f"    Significant: {comparison['is_significant']}")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Add rank within each task (by delta_acc, most negative = rank 1)
    df["rank"] = df.groupby("task")["delta_acc"].rank(method="dense", ascending=True)

    # Save results
    output_path = output_dir / "layer_ablation.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

    # Identify critical layers
    critical_layers = identify_critical_layers(
        df,
        threshold=-0.20,  # Delta accuracy < -0.20
        min_tasks=2,  # Affect at least 2 tasks
    )

    summary_list: list[dict[str, Any]] = [
        {
            "layer_idx": int(layer),
            "affected_tasks": df[(df["layer_idx"] == layer) & (df["delta_acc"] < -0.20)]["task"].tolist(),
            "avg_delta_acc": float(df[df["layer_idx"] == layer]["delta_acc"].mean()),
            "max_delta_acc": float(df[df["layer_idx"] == layer]["delta_acc"].min()),
        }
        for layer in critical_layers
    ]

    critical_info: dict[str, Any] = {
        "critical_layers": critical_layers,
        "threshold": -0.20,
        "min_tasks": 2,
        "summary": summary_list,
    }

    # Save critical layers info
    critical_path = output_dir / "critical_layers.json"
    with open(critical_path, "w") as f:
        json.dump(critical_info, f, indent=2)
    print(f"\nSaved critical layers to: {critical_path}")

    print("\n" + "=" * 80)
    print("Layer Ablation Summary:")
    print("=" * 80)
    print(f"Critical layers: {critical_layers}")
    for info in summary_list:
        print(f"\nLayer {info['layer_idx']}:")
        print(f"  Affected tasks: {', '.join(info['affected_tasks'])}")
        print(f"  Avg delta acc: {info['avg_delta_acc']:+.4f}")
        print(f"  Max delta acc: {info['max_delta_acc']:+.4f}")

    return df


def identify_critical_layers(
    df: pd.DataFrame,
    threshold: float = -0.20,
    min_tasks: int = 2,
) -> list[int]:
    """
    Identify critical layers based on performance degradation.

    A layer is considered critical if:
    - It causes significant performance drop (delta_acc < threshold)
    - It affects multiple tasks (>= min_tasks)

    Args:
        df: DataFrame with layer ablation results
        threshold: Delta accuracy threshold (negative value)
        min_tasks: Minimum number of tasks affected

    Returns:
        List of critical layer indices
    """
    # Count tasks affected by each layer
    affected_counts = df[df["delta_acc"] < threshold].groupby("layer_idx")["task"].count().reset_index()
    affected_counts.columns = ["layer_idx", "n_tasks_affected"]

    # Filter layers that affect enough tasks
    critical = affected_counts[affected_counts["n_tasks_affected"] >= min_tasks]

    # Get average performance drop for each critical layer
    avg_delta = df[df["layer_idx"].isin(critical["layer_idx"])].groupby("layer_idx")["delta_acc"].mean().reset_index()

    # Sort by average delta (most negative first)
    avg_delta = avg_delta.sort_values("delta_acc")

    critical_layers = avg_delta["layer_idx"].tolist()

    return critical_layers
