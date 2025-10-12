"""Script for cross-condition probing experiment.

This script:
1. Loads pre-extracted features from Image ON and Image OFF experiments
2. Runs cross-condition probing for all tasks and layers
3. Generates cross-condition matrices and plots
"""

import json
from pathlib import Path
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.schema import Config
from src.probing.cross_condition import (
    cross_condition_probe_all_layers,
    summarize_cross_condition_results,
)
from src.utils import get_experiment_output_dir
from src.visualization import (
    plot_cross_condition_gaps,
    plot_cross_condition_matrix,
    plot_cross_condition_prober_accuracy,
)

# Minimum feature dimension for probing
MIN_FEATURE_DIM = 2


def load_features_from_dir(
    task_dir: Path,
    layer_names: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Load features and labels from a task directory.

    Args:
        task_dir: Task directory containing features_*.npy and labels.npy
        layer_names: List of layer names to load (default: all)

    Returns:
        Tuple of (features dict, labels array)
    """
    labels_path = task_dir / "labels.npy"
    if not labels_path.exists():
        error_msg = f"labels.npy not found in {task_dir}"
        raise FileNotFoundError(error_msg)

    labels = np.load(labels_path)
    features: dict[str, np.ndarray] = {}

    feature_files = sorted(task_dir.glob("features_*.npy"))
    for fp in feature_files:
        layer_name = fp.stem.replace("features_", "")

        # Skip output-averaged features (we use full features for probing)
        if layer_name.endswith("_outavg"):
            continue

        if layer_names is not None and layer_name not in layer_names:
            continue

        feat = np.load(fp)
        if feat.shape[0] == labels.shape[0] and feat.ndim == MIN_FEATURE_DIM:
            features[layer_name] = feat
        else:
            msg = f"[WARN] Skipping {layer_name}: shape {feat.shape} incompatible with labels {labels.shape}"
            print(msg)

    return features, labels


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Run cross-condition probing experiment.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("Cross-condition Probing Experiment")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Convert to Pydantic model for validation
    config = Config(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

    # Determine experiment directory structure
    experiment_name = "cross_condition"
    cross_cond_root = get_experiment_output_dir(config.output.results_root, experiment_name, config.model.model_id)
    cross_cond_root.mkdir(parents=True, exist_ok=True)

    # For backward compatibility: look for features in comparison experiment
    comparison_root = get_experiment_output_dir(config.output.results_root, "comparison", config.model.model_id)

    print("\nLooking for features in comparison experiment:")
    print(f"  Comparison dir: {comparison_root}")
    print(f"\nOutput directory: {cross_cond_root}")

    # =========================================================================
    # Run cross-condition probing for all tasks
    # =========================================================================
    all_results: dict[str, dict] = {}
    summary_rows: list[dict] = []

    for task in config.experiment.tasks:
        print("\n" + "=" * 80)
        print(f"TASK: {task}")
        print("=" * 80)

        # Find directories (look in comparison experiment)
        dir_imageon = comparison_root / f"{task}_imageon"
        dir_imageoff = comparison_root / f"{task}_imageoff"

        if not dir_imageon.exists():
            print(f"[SKIP] Directory not found: {dir_imageon}")
            continue

        if not dir_imageoff.exists():
            print(f"[SKIP] Directory not found: {dir_imageoff}")
            continue

        # Load features
        print("\nLoading features from:")
        print(f"  Image ON:  {dir_imageon}")
        print(f"  Image OFF: {dir_imageoff}")

        try:
            features_imageon, labels_imageon = load_features_from_dir(dir_imageon)
            features_imageoff, labels_imageoff = load_features_from_dir(dir_imageoff)
        except Exception as e:
            print(f"[ERROR] Failed to load features: {e}")
            continue

        # Verify labels match
        if not np.array_equal(labels_imageon, labels_imageoff):
            print(f"[ERROR] Labels mismatch between Image ON and Image OFF for task {task}")
            continue

        print(f"\nLoaded features for {len(features_imageon)} layers")
        print(f"Number of samples: {len(labels_imageon)}")
        common_layers = sorted(set(features_imageon.keys()) & set(features_imageoff.keys()))
        print(f"Common layers: {common_layers[:10]}...")

        # Run cross-condition probing
        print("\nRunning cross-condition probing...")
        results = cross_condition_probe_all_layers(
            features_condA=features_imageon,
            features_condB=features_imageoff,
            labels=labels_imageon,
            train_ratio=0.8,
            seed=config.probe.seed,
            max_iter=config.probe.max_iter,
            C=config.probe.C,
            solver=config.probe.solver,
        )

        all_results[task] = results

        # Save results
        task_output_dir = get_experiment_output_dir(config.output.results_root, experiment_name, config.model.model_id, task)
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        results_json = {}
        for layer, metrics in results.items():
            results_json[layer] = {
                "imageon_to_imageoff": metrics["A_to_B"],
                "imageoff_to_imageon": metrics["B_to_A"],
            }

        output_path = task_output_dir / "cross_condition_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nSaved results to: {output_path}")

        # Infer layer order from loaded features
        vision_keys = [k for k in ["v_enc", "v_proj"] if k in features_imageon]
        llm_layers = sorted([k for k in features_imageon if k.startswith("l") and k[:3].replace("l", "").isdigit()])
        layer_order = vision_keys + llm_layers

        # Summarize results
        summary = summarize_cross_condition_results(results, layer_order=layer_order)

        # Save summary arrays
        summary_path = task_output_dir / "cross_condition_summary.npz"
        np.savez(
            str(summary_path),
            layers=summary["layers"],
            A_same_acc=summary["A_same_acc"],
            A_cross_acc=summary["A_cross_acc"],
            A_gap=summary["A_gap"],
            B_same_acc=summary["B_same_acc"],
            B_cross_acc=summary["B_cross_acc"],
            B_gap=summary["B_gap"],
        )
        print(f"Saved summary to: {summary_path}")

        # Print summary statistics
        print("\nSummary Statistics:")
        print("-" * 80)
        print(f"{'Direction':<25} {'Mean Gap':<15} {'Max Gap':<15} {'Max Gap Layer':<15}")
        print("-" * 80)

        # ImageON → ImageOFF
        valid_mask = ~np.isnan(summary["A_gap"])
        if valid_mask.any():
            mean_gap = np.nanmean(summary["A_gap"])
            max_gap_idx = np.nanargmax(summary["A_gap"])
            max_gap = summary["A_gap"][max_gap_idx]
            max_gap_layer = summary["layers"][max_gap_idx]
            print(f"{'ImageON → ImageOFF':<25} {mean_gap:.4f}{'':>9} {max_gap:.4f}{'':>9} {max_gap_layer:<15}")

        # ImageOFF → ImageON
        valid_mask = ~np.isnan(summary["B_gap"])
        if valid_mask.any():
            mean_gap = np.nanmean(summary["B_gap"])
            max_gap_idx = np.nanargmax(summary["B_gap"])
            max_gap = summary["B_gap"][max_gap_idx]
            max_gap_layer = summary["layers"][max_gap_idx]
            print(f"{'ImageOFF → ImageON':<25} {mean_gap:.4f}{'':>9} {max_gap:.4f}{'':>9} {max_gap_layer:<15}")

        print("-" * 80)

        # Add to summary rows
        summary_rows.append(
            {
                "task": task,
                "imageon_to_imageoff_mean_gap": np.nanmean(summary["A_gap"]),
                "imageon_to_imageoff_max_gap": np.nanmax(summary["A_gap"]),
                "imageoff_to_imageon_mean_gap": np.nanmean(summary["B_gap"]),
                "imageoff_to_imageon_max_gap": np.nanmax(summary["B_gap"]),
                "n_samples": len(labels_imageon),
                "n_layers": int(valid_mask.sum()),
            }
        )

    # =========================================================================
    # Generate overall summary
    # =========================================================================
    if summary_rows:
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)

        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))

        # Save summary
        summary_df.to_csv(cross_cond_root / "summary.csv", index=False)
        print(f"\nSaved overall summary to: {cross_cond_root / 'summary.csv'}")

    # =========================================================================
    # Generate plots (if enabled)
    # =========================================================================
    if config.output.save_plots:
        print("\n" + "=" * 80)
        print("Generating plots...")
        print("=" * 80)

        try:
            for task, task_results in all_results.items():
                task_output_dir = cross_cond_root / task
                summary_path = task_output_dir / "cross_condition_summary.npz"

                # Create plots directory
                plots_dir = task_output_dir / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)

                if summary_path.exists():
                    summary_data = np.load(summary_path)

                    # Plot accuracy gaps
                    gap_plot_path = plots_dir / "cross_condition_gaps.png"
                    plot_cross_condition_gaps(
                        layers=summary_data["layers"],
                        gap_A_to_B=summary_data["A_gap"],
                        gap_B_to_A=summary_data["B_gap"],
                        task=task,
                        title_suffix=f"{config.model.model_id}",
                        output_path=str(gap_plot_path),
                    )

                    # Plot prober accuracies
                    prober_acc_plot_path = plots_dir / "cross_condition_prober_accuracy.png"
                    plot_cross_condition_prober_accuracy(
                        layers=summary_data["layers"],
                        A_same_acc=summary_data["A_same_acc"],
                        A_cross_acc=summary_data["A_cross_acc"],
                        B_cross_acc=summary_data["B_cross_acc"],
                        B_same_acc=summary_data["B_same_acc"],
                        task=task,
                        title_suffix=f"{config.model.model_id}",
                        output_path=str(prober_acc_plot_path),
                    )

                    # Plot cross-condition matrix for selected layers
                    selected_layers = [k for k in task_results if k.startswith("l") and k[:3].replace("l", "").isdigit()]
                    for layer in selected_layers:
                        if layer in task_results:
                            matrix_plot_path = plots_dir / f"cross_condition_matrix_{layer}.png"
                            plot_cross_condition_matrix(
                                task=task,
                                layer=layer,
                                metrics=task_results[layer],
                                output_path=str(matrix_plot_path),
                            )

        except Exception as e:
            print(f"[WARN] Failed to generate plots: {e}")
            print("You can generate plots later using the saved data.")

    print("\n" + "=" * 80)
    print("Cross-condition probing experiment completed!")
    print("=" * 80)
    print(f"\nResults saved to: {cross_cond_root}")


if __name__ == "__main__":
    main()
