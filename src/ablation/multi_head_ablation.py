"""Progressive multi-head ablation experiment.

This experiment tests the cooperative effects of multiple attention heads
by progressively ablating increasing numbers of heads and measuring
performance degradation.
"""

import json
import logging
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

from src.data import HuggingFaceDataset
from src.utils.paths import get_model_short_name

from .evaluator import AblationEvaluator
from .hooks import AblationHookManager

logger = logging.getLogger(__name__)


def run_progressive_multi_head_ablation(
    model: Any,
    processor: Any,
    config: Any,
    target_layers: list[int] | None = None,
    tasks: list[str] | None = None,
    output_dir: str | Path | None = None,
    device: str = "cuda",
    num_heads: int = 28,
    n_heads_values: list[int] | None = None,
    n_trials: int = 10,
    random_seed: int = 42,
    show_progress: bool = True,
    model_id: str | None = None,
) -> dict[str, Any]:
    """
    Run progressive multi-head ablation experiment.

    This experiment ablates increasing numbers of heads (1, 2, 4, 8, ...)
    and measures performance degradation to test cooperative effects.

    Args:
        model: VLM model
        processor: Model processor
        config: Experiment configuration
        target_layers: List of layers to analyze (default: [14, 15, 16, 17])
        tasks: List of tasks to evaluate (default: use config.experiment.tasks)
        output_dir: Output directory (default: config.output.results_root/ablation/multi_head/{model_id})
        device: Device to run on
        num_heads: Total number of attention heads per layer
        n_heads_values: List of n_heads to ablate (default: [1, 2, 4, 8, 16, 24, 28])
        n_trials: Number of random trials per n_heads
        random_seed: Random seed for reproducibility
        show_progress: Show progress bars
        model_id: Model identifier for organizing results (default: from config)

    Returns:
        Dictionary with experimental results and analysis
    """
    # Setup
    if tasks is None:
        tasks = config.experiment.tasks

    if n_heads_values is None:
        n_heads_values = [1, 2, 4, 8, 16, 24, 28]

    if target_layers is None:
        target_layers = [14, 15, 16, 17]

    # Get model short name for directory structure
    if model_id is None:
        model_id = get_model_short_name(config.model.model_id)

    if output_dir is None:
        output_dir = Path(config.output.results_root) / "ablation" / "multi_head" / model_id
    else:
        output_dir = Path(output_dir)

    # Create subdirectories
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    for dir_path in [figures_dir, tables_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("\nOutput directory structure:")
    print(f"  Base: {output_dir}")
    print(f"  Figures: {figures_dir}")
    print(f"  Tables: {tables_dir}")

    # Initialize evaluator
    evaluator = AblationEvaluator(
        model=model,
        processor=processor,
        device=device,
        batch_size=config.batch_size,
        num_heads=num_heads,
        evaluation_position=config.ablation.get("evaluation_position", "answer_head") if hasattr(config, "ablation") else "answer_head",
    )

    # Get dataset name
    dataset_name = config.dataset.hf_dataset
    split = config.dataset.hf_split

    # Set random seed
    np.random.seed(random_seed)

    # Phase 1: Baseline measurement
    print("\n" + "=" * 80)
    print("Phase 1: Baseline Measurement")
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

    # Phase 2: Progressive Multi-Head Ablation
    print("\n" + "=" * 80)
    print("Phase 2: Progressive Multi-Head Ablation")
    print("=" * 80)
    print(f"Target layers: {target_layers}")
    print(f"n_heads values: {n_heads_values}")
    print(f"n_trials per n_heads: {n_trials}")
    print(f"Random seed: {random_seed}")

    results: dict[str, Any] = {
        "baseline": baseline_results,
        "ablation": {},
        "config": {
            "target_layers": target_layers,
            "n_heads_values": n_heads_values,
            "n_trials": n_trials,
            "random_seed": random_seed,
            "num_heads": num_heads,
            "tasks": tasks,
        },
    }

    total_iterations = len(target_layers) * len(n_heads_values) * n_trials * len(tasks)
    with tqdm(total=total_iterations, desc="Multi-head ablation", disable=not show_progress) as pbar:
        for layer_idx in target_layers:
            print(f"\n--- Layer {layer_idx} ---")
            layer_results = {}

            for n_heads in n_heads_values:
                if n_heads > num_heads:
                    print(f"  Skipping n_heads={n_heads} (exceeds total {num_heads})")
                    continue

                print(f"\n  Ablating {n_heads} heads (Layer {layer_idx})...")
                trial_results = []

                for trial_idx in range(n_trials):
                    pbar.set_description(f"L{layer_idx} n={n_heads} trial {trial_idx + 1}/{n_trials}")

                    # Random sample of heads
                    sampled_heads = np.random.choice(
                        num_heads,
                        size=n_heads,
                        replace=False,
                    ).tolist()

                    # Register hook for multiple heads
                    hook_manager = AblationHookManager()
                    hook_manager.register_multi_head_hook(
                        model=model,
                        layer_idx=layer_idx,
                        head_indices=sampled_heads,
                        ablation_type=config.ablation.get("method", "zero") if hasattr(config, "ablation") else "zero",
                        num_heads=num_heads,
                    )

                    try:
                        # Evaluate all tasks with hook already registered
                        task_accuracies = {}
                        for task in tasks:
                            # Load dataset
                            try:
                                dataset = HuggingFaceDataset(
                                    dataset_name=dataset_name,
                                    split=split if split != "auto" else task,
                                    task=None if split == task else task,
                                )
                            except Exception:
                                # Fallback
                                dataset = HuggingFaceDataset(
                                    dataset_name=dataset_name,
                                    split="train",
                                    task=task,
                                )

                            # Evaluate with hook already registered (pass None to avoid double registration)
                            ablated = evaluator.evaluate_with_ablation(
                                dataset=dataset,
                                layer_idx=None,  # Hook already registered
                                head_idx=None,
                                ablation_type=config.ablation.get("method", "zero") if hasattr(config, "ablation") else "zero",
                                show_progress=False,
                            )
                            task_accuracies[task] = ablated["accuracy"]
                            pbar.update(1)

                        # Store trial results
                        trial_results.append(
                            {
                                "sampled_heads": sampled_heads,
                                "accuracies": task_accuracies,
                            }
                        )

                    finally:
                        # Remove hook
                        hook_manager.remove_all_hooks()

                        # Verify hooks are removed
                        try:
                            if hasattr(model, "model") and hasattr(model.model, "language_model"):
                                layers = model.model.language_model.layers
                            elif hasattr(model, "language_model"):
                                layers = model.language_model.layers
                            else:
                                layers = None

                            if layers is not None:
                                target_layer = layers[layer_idx]
                                if hasattr(target_layer, "self_attn") and hasattr(target_layer.self_attn, "o_proj"):
                                    o_proj_module = target_layer.self_attn.o_proj
                                    # Check if hooks are registered (using getattr to avoid private member access)
                                    hooks = getattr(o_proj_module, "_forward_pre_hooks", {})
                                    num_hooks = len(hooks)
                                    if num_hooks > 0:
                                        logger.warning("  WARNING: %d hooks still registered after removal!", num_hooks)

                        except Exception as e:
                            logger.debug("Could not verify hook removal: %s", e)

                layer_results[n_heads] = trial_results

            results["ablation"][str(layer_idx)] = layer_results

    # Phase 3: Statistical Analysis
    print("\n" + "=" * 80)
    print("Phase 3: Statistical Analysis")
    print("=" * 80)

    analysis = analyze_multi_head_results(results, baseline_results)

    # Save results
    print("\nSaving results...")
    with open(output_dir / "raw_results.pkl", "wb") as f:
        pickle.dump(results, f)

    with open(output_dir / "analysis.json", "w") as f:
        # Convert numpy types to native Python types for JSON
        analysis_json = json.loads(json.dumps(analysis, default=str))
        json.dump(analysis_json, f, indent=2)

    # Save summary tables
    save_summary_tables(analysis, baseline_results, tables_dir, tasks, n_heads_values)

    # Generate summary report
    generate_summary_report(
        baseline_results,
        analysis,
        output_dir / "summary_report.txt",
        tasks,
        n_heads_values,
    )

    print(f"\nResults saved to: {output_dir}")

    return {
        "results": results,
        "analysis": analysis,
        "output_dir": output_dir,
    }


def analyze_multi_head_results(
    results: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """
    Analyze multi-head ablation results.

    Computes statistics across trials, effect sizes, and significance tests.

    Args:
        results: Experimental results dictionary
        baseline: Baseline results dictionary

    Returns:
        Analysis dictionary with statistics
    """
    ablation_data = results["ablation"]
    tasks = results["config"]["tasks"]

    analysis = {}

    for layer_idx, layer_results in ablation_data.items():
        layer_analysis = {}

        for n_heads, trials in layer_results.items():
            # Aggregate across trials
            task_accuracies_per_trial: dict[str, list[float]] = {task: [] for task in tasks}

            for trial in trials:
                for task, acc in trial["accuracies"].items():
                    task_accuracies_per_trial[task].append(acc)

            # Compute statistics per task
            task_stats = {}
            for task, accuracies in task_accuracies_per_trial.items():
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)

                baseline_acc = baseline[task]["accuracy"]

                # Effect size (Cohen's d)
                pooled_std = std_acc  # Simplified (baseline std ≈ 0)
                if pooled_std > 0:
                    cohens_d = (baseline_acc - mean_acc) / pooled_std
                else:
                    cohens_d = 0.0

                # T-test (one-sample, test if different from baseline)
                _t_stat, p_value = stats.ttest_1samp(accuracies, baseline_acc)

                task_stats[task] = {
                    "mean": float(mean_acc),
                    "std": float(std_acc),
                    "baseline": float(baseline_acc),
                    "delta": float(mean_acc - baseline_acc),
                    "cohens_d": float(cohens_d),
                    "p_value": float(p_value),
                    "significant": p_value < 0.001,  # Bonferroni corrected threshold
                }

            # Overall statistics
            overall_mean = np.mean([s["mean"] for s in task_stats.values()])
            overall_delta = np.mean([s["delta"] for s in task_stats.values()])

            layer_analysis[n_heads] = {
                "task_stats": task_stats,
                "overall_mean": float(overall_mean),
                "overall_delta": float(overall_delta),
                "n_significant_tasks": sum(1 for s in task_stats.values() if s["significant"]),
            }

        analysis[layer_idx] = layer_analysis

    return analysis


def save_summary_tables(
    analysis: dict[str, Any],
    baseline: dict[str, Any],  # noqa: ARG001
    tables_dir: Path,
    tasks: list[str],  # noqa: ARG001
    n_heads_values: list[int],  # noqa: ARG001
) -> None:
    """Save summary statistics tables."""
    # Summary statistics table
    summary_data = []
    for layer_idx, layer_analysis in analysis.items():
        for n_heads, stats_dict in layer_analysis.items():
            summary_data.append(
                {
                    "layer": int(layer_idx),
                    "n_heads": int(n_heads),
                    "overall_mean": stats_dict["overall_mean"],
                    "overall_delta": stats_dict["overall_delta"],
                    "n_significant_tasks": stats_dict["n_significant_tasks"],
                }
            )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(tables_dir / "summary_statistics.csv", index=False)

    # Per-task breakdown
    task_data = []
    for layer_idx, layer_analysis in analysis.items():
        for n_heads, stats_dict in layer_analysis.items():
            for task, task_stat in stats_dict["task_stats"].items():
                task_data.append(
                    {
                        "layer": int(layer_idx),
                        "n_heads": int(n_heads),
                        "task": task,
                        "baseline_acc": task_stat["baseline"],
                        "mean_acc": task_stat["mean"],
                        "std_acc": task_stat["std"],
                        "delta_acc": task_stat["delta"],
                        "cohens_d": task_stat["cohens_d"],
                        "p_value": task_stat["p_value"],
                        "significant": task_stat["significant"],
                    }
                )

    task_df = pd.DataFrame(task_data)
    task_df.to_csv(tables_dir / "per_task_breakdown.csv", index=False)

    print(f"  Saved summary tables to: {tables_dir}")


def generate_summary_report(
    baseline: dict[str, Any],
    analysis: dict[str, Any],
    output_path: Path,
    tasks: list[str],
    n_heads_values: list[int],  # noqa: ARG001
) -> None:
    """Generate text summary report."""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Multi-Head Ablation Experiment Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("Baseline Accuracies:\n")
        f.write("-" * 80 + "\n")
        f.writelines(f"  {task:15s}: {result['accuracy']:.4f}\n" for task, result in baseline.items())
        f.write("\n")

        for layer_idx, layer_analysis in analysis.items():
            f.write(f"\nLayer {layer_idx} Analysis:\n")
            f.write("=" * 80 + "\n")

            for n_heads in sorted(layer_analysis.keys()):
                stats_dict = layer_analysis[n_heads]
                f.write(f"\n  n_heads = {n_heads}:\n")
                f.write(f"    Overall mean accuracy: {stats_dict['overall_mean']:.4f}\n")
                f.write(f"    Overall delta: {stats_dict['overall_delta']:.4f}\n")
                f.write(f"    Significant tasks: {stats_dict['n_significant_tasks']}/{len(tasks)}\n")

                # Show task-specific results
                f.write("\n    Per-task breakdown:\n")
                for task in tasks:
                    task_stat = stats_dict["task_stats"][task]
                    sig_marker = "***" if task_stat["significant"] else ""
                    f.write(
                        f"      {task:15s}: {task_stat['mean']:.4f} "
                        f"(Δ={task_stat['delta']:+.4f}, p={task_stat['p_value']:.2e}) {sig_marker}\n"
                    )

        f.write("\n" + "=" * 80 + "\n")

    print(f"  Saved summary report to: {output_path}")
