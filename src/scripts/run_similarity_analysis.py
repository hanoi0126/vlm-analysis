"""Script for representation similarity analysis.

This script:
1. Loads pre-extracted features from Image ON and Image OFF experiments
2. Computes similarity metrics (CKA, cosine, etc.) for all layers
3. Generates similarity plots and 2D visualizations
"""

import json
from pathlib import Path
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.similarity import compute_similarity_all_layers
from src.config.schema import Config
from src.utils import get_experiment_output_dir
from src.visualization.similarity_plots import (
    plot_2d_comparison,
    plot_layer_trajectory,
    plot_similarity_curves,
)


def load_features_from_dir(
    task_dir: Path,
    layer_names: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load features and labels from a task directory."""
    labels_path = task_dir / "labels.npy"
    if not labels_path.exists():
        msg = f"labels.npy not found in {task_dir}"
        raise FileNotFoundError(msg)

    labels = np.load(labels_path)
    features: dict[str, np.ndarray] = {}

    feature_files = sorted(task_dir.glob("features_*.npy"))
    for fp in feature_files:
        layer_name = fp.stem.replace("features_", "")

        # Skip output-averaged features
        if layer_name.endswith("_outavg"):
            continue

        if layer_names is not None and layer_name not in layer_names:
            continue

        feat = np.load(fp)
        if feat.shape[0] == labels.shape[0] and feat.ndim == 2:
            features[layer_name] = feat

    return features, labels


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run representation similarity analysis."""
    print("=" * 80)
    print("Representation Similarity Analysis")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Convert to Pydantic model
    config = Config(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

    # Determine experiment directory structure
    experiment_name = "similarity"
    similarity_root = get_experiment_output_dir(config.output.results_root, experiment_name, config.model.model_id)
    similarity_root.mkdir(parents=True, exist_ok=True)

    # For backward compatibility: look for features in comparison experiment
    comparison_root = get_experiment_output_dir(config.output.results_root, "comparison", config.model.model_id)

    print("\nLooking for features in comparison experiment:")
    print(f"  Comparison dir: {comparison_root}")
    print(f"\nOutput directory: {similarity_root}")

    # Layer order
    layer_order = ["pre", "post"] + [f"l{i:02d}" for i in range(36)]

    # Summary rows
    summary_rows: list[dict] = []

    # =========================================================================
    # Compute similarities for all tasks
    # =========================================================================
    for task in config.experiment.tasks:
        print("\n" + "=" * 80)
        print(f"TASK: {task}")
        print("=" * 80)

        # Find directories (look in comparison experiment)
        dir_imageon = comparison_root / f"{task}_imageon"
        dir_imageoff = comparison_root / f"{task}_imageoff"

        if not dir_imageon.exists() or not dir_imageoff.exists():
            print(f"[SKIP] Directories not found for task {task}")
            print(f"  Expected Image ON:  {dir_imageon}")
            print(f"  Expected Image OFF: {dir_imageoff}")
            continue

        # Load features
        print("\nLoading features...")
        try:
            features_imageon, labels_imageon = load_features_from_dir(dir_imageon)
            features_imageoff, labels_imageoff = load_features_from_dir(dir_imageoff)
        except Exception as e:
            print(f"[ERROR] Failed to load features: {e}")
            continue

        if not np.array_equal(labels_imageon, labels_imageoff):
            print(f"[ERROR] Labels mismatch for task {task}")
            continue

        print(f"Loaded {len(features_imageon)} layers, {len(labels_imageon)} samples")

        # Compute similarities
        print("\nComputing similarity metrics...")
        similarities = compute_similarity_all_layers(
            features_a=features_imageon,
            features_b=features_imageoff,
            methods=["cka", "cosine"],
            layer_order=layer_order,
        )

        # Save results
        task_output_dir = get_experiment_output_dir(config.output.results_root, experiment_name, config.model.model_id, task)
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # Save to JSON
        output_path = task_output_dir / "similarity_metrics.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(similarities, f, indent=2)
        print(f"\nSaved results to: {output_path}")

        # Print summary
        print("\nSummary Statistics:")
        print("-" * 80)
        cka_values = [v.get("cka", np.nan) for v in similarities.values()]
        cka_mean = np.nanmean(cka_values)
        cka_min = np.nanmin(cka_values)
        cka_max = np.nanmax(cka_values)
        print(f"CKA: mean={cka_mean:.4f}, min={cka_min:.4f}, max={cka_max:.4f}")

        cosine_values = [v.get("cosine", np.nan) for v in similarities.values()]
        cosine_mean = np.nanmean(cosine_values)
        cosine_min = np.nanmin(cosine_values)
        cosine_max = np.nanmax(cosine_values)
        print(f"Cosine: mean={cosine_mean:.4f}, min={cosine_min:.4f}, max={cosine_max:.4f}")
        print("-" * 80)

        # Add to summary
        summary_rows.append(
            {
                "task": task,
                "cka_mean": cka_mean,
                "cka_min": cka_min,
                "cka_max": cka_max,
                "cosine_mean": cosine_mean,
                "cosine_min": cosine_min,
                "cosine_max": cosine_max,
                "n_samples": len(labels_imageon),
                "n_layers": len(similarities),
            }
        )

        # =====================================================================
        # Generate plots
        # =====================================================================
        if config.output.save_plots:
            plots_dir = task_output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # 1. Similarity curves
            print("\nGenerating similarity curves...")
            plot_similarity_curves(
                layers=layer_order,
                similarities=similarities,
                task=task,
                metrics=["cka", "cosine"],
                title_suffix=f"{config.model.model_id}",
                output_path=str(plots_dir / "similarity_curves.png"),
            )

            # 2. 2D comparison for selected layers
            print("Generating 2D comparisons...")
            selected_layers = ["l00", "l12", "l24", "l35"]
            for layer in selected_layers:
                if layer in features_imageon and layer in features_imageoff:
                    plot_2d_comparison(
                        features_a=features_imageon[layer],
                        features_b=features_imageoff[layer],
                        labels=labels_imageon,
                        layer=layer,
                        task=task,
                        method="pca",
                        output_path=str(plots_dir / f"2d_comparison_{layer}.png"),
                    )

            # 3. Layer trajectory
            print("Generating layer trajectory...")
            trajectory_layers = ["l00", "l06", "l12", "l18", "l24", "l30", "l35"]
            plot_layer_trajectory(
                all_features_a=features_imageon,
                all_features_b=features_imageoff,
                labels=labels_imageon,
                selected_layers=trajectory_layers,
                task=task,
                method="pca",
                sample_size=100,
                output_path=str(plots_dir / "layer_trajectory.png"),
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
        summary_df.to_csv(similarity_root / "summary.csv", index=False)
        print(f"\nSaved overall summary to: {similarity_root / 'summary.csv'}")

    print("\n" + "=" * 80)
    print("Similarity analysis completed!")
    print("=" * 80)
    print(f"\nResults saved to: {similarity_root}")


if __name__ == "__main__":
    main()
