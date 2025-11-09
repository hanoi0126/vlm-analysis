"""Script for VLM vs LLM PRAG comparison experiment."""

import json
from pathlib import Path
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Any

from src.config.schema import Config
from src.data import HuggingFaceDataset
from src.models.registry import create_extractor
from src.probing.prag_analysis import analyze_prag_with_dataset_classes
from src.probing.prag_statistics import PRAGStatistics
from src.probing.runner import run_extract_probe_decode
from src.utils import get_experiment_output_dir


def convert_to_json_serializable(obj: Any) -> Any:  # noqa: PLR0911
    """
    Convert numpy/pandas types to JSON-serializable Python types.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable object
    """
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    if pd.isna(obj):
        return None
    return obj


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Run VLM vs LLM PRAG comparison experiment.

    This script:
    1. Runs feature extraction WITH images (VLM condition)
    2. Runs feature extraction WITHOUT images (LLM condition)
    3. Computes PRAG for both conditions
    4. Performs statistical comparison
    5. Generates results and visualizations

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("VLM vs LLM PRAG Comparison Experiment")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Convert to Pydantic model for validation
    config = Config(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

    # Create feature extractor
    print(f"\nLoading model: {config.model.model_id}")
    extractor = create_extractor(config.model)
    print("Model loaded successfully")
    print(f"Tap points: {extractor.get_tap_points()[:5]}... ({len(extractor.get_tap_points())} total)")

    # Set up experiment output directory
    experiment_name = "prag_comparison"
    prag_root = get_experiment_output_dir(config.output.results_root, experiment_name, config.model.model_id)
    print(f"\nExperiment output directory: {prag_root}")

    # Determine target layer
    target_layer = getattr(config.prag, "target_layer", "best") if hasattr(config, "prag") else "best"
    if target_layer == "best":
        # Use a middle layer as default (e.g., l19)
        target_layer = "l19"
    print(f"Target layer for PRAG: {target_layer}")

    # =========================================================================
    # Part 1: Run with images (VLM condition)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: Running WITH images (VLM condition)")
    print("=" * 80)

    config.output.results_root = prag_root

    print(f"\nExtracting features for tasks: {config.experiment.tasks}")
    summary_vlm = run_extract_probe_decode(
        extractor=extractor,
        config=config,
        use_image=True,
        show_progress=True,
        condition_suffix="_vlm",
    )
    print("\nExtraction summary (VLM):")
    print(summary_vlm)

    # =========================================================================
    # Part 2: Run without images (LLM condition)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: Running WITHOUT images (LLM condition)")
    print("=" * 80)

    print(f"\nExtracting features for tasks: {config.experiment.tasks}")
    summary_llm = run_extract_probe_decode(
        extractor=extractor,
        config=config,
        use_image=False,
        show_progress=True,
        condition_suffix="_llm",
    )
    print("\nExtraction summary (LLM):")
    print(summary_llm)

    # =========================================================================
    # Part 3: Compute PRAG for both conditions
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: Computing PRAG for both conditions")
    print("=" * 80)

    prag_results = []
    stats_obj = PRAGStatistics()

    for task in config.experiment.tasks:
        print(f"\nProcessing task: {task}")

        # Load datasets
        actual_split = task if config.dataset.hf_split == "auto" else config.dataset.hf_split
        filter_task = None if actual_split == task else task

        ds_vlm = HuggingFaceDataset(
            dataset_name=config.dataset.hf_dataset,  # type: ignore[arg-type]
            split=actual_split,
            subset=config.dataset.hf_subset,
            task=filter_task,
            cache_dir=None,
        )

        ds_llm = HuggingFaceDataset(
            dataset_name=config.dataset.hf_dataset,  # type: ignore[arg-type]
            split=actual_split,
            subset=config.dataset.hf_subset,
            task=filter_task,
            cache_dir=None,
        )

        # Load features
        task_dir_vlm = prag_root / f"{task}_vlm"
        task_dir_llm = prag_root / f"{task}_llm"

        features_path_vlm = task_dir_vlm / f"features_{target_layer}.npy"
        features_path_llm = task_dir_llm / f"features_{target_layer}.npy"
        labels_path_vlm = task_dir_vlm / "labels.npy"
        labels_path_llm = task_dir_llm / "labels.npy"

        if not all(
            [
                features_path_vlm.exists(),
                features_path_llm.exists(),
                labels_path_vlm.exists(),
                labels_path_llm.exists(),
            ]
        ):
            print(f"[WARN] Missing files for {task}, skipping")
            continue

        features_vlm = np.load(features_path_vlm)
        features_llm = np.load(features_path_llm)
        labels_vlm = np.load(labels_path_vlm)
        labels_llm = np.load(labels_path_llm)

        # Compute PRAG for VLM
        try:
            prag_vlm_result = analyze_prag_with_dataset_classes(
                extractor=extractor,
                dataset=ds_vlm,
                features=features_vlm,
                labels=labels_vlm,
                layer_name=target_layer,
                max_iter=config.probe.max_iter,
                C=config.probe.C,
                solver=config.probe.solver,
            )
            prag_vlm = prag_vlm_result["prag"]["prag_mean"]
        except Exception as e:
            print(f"[ERROR] Failed to compute PRAG for VLM {task}: {e}")
            prag_vlm = np.nan
            prag_vlm_result = None

        # Compute PRAG for LLM
        try:
            prag_llm_result = analyze_prag_with_dataset_classes(
                extractor=extractor,
                dataset=ds_llm,
                features=features_llm,
                labels=labels_llm,
                layer_name=target_layer,
                max_iter=config.probe.max_iter,
                C=config.probe.C,
                solver=config.probe.solver,
            )
            prag_llm = prag_llm_result["prag"]["prag_mean"]
        except Exception as e:
            print(f"[ERROR] Failed to compute PRAG for LLM {task}: {e}")
            prag_llm = np.nan
            prag_llm_result = None

        # Statistical test
        if not (np.isnan(prag_vlm) or np.isnan(prag_llm)):
            # For single task, we need multiple samples - use per-class PRAG
            if prag_vlm_result is not None and prag_llm_result is not None:
                prag_vlm_per_class = prag_vlm_result["prag"]["prag_per_class"]
                prag_llm_per_class = prag_llm_result["prag"]["prag_per_class"]

                if len(prag_vlm_per_class) == len(prag_llm_per_class):
                    test_result = stats_obj.test_vlm_vs_llm(prag_vlm_per_class, prag_llm_per_class)
                else:
                    test_result = None
            else:
                test_result = None
        else:
            test_result = None

        prag_results.append(
            {
                "task": task,
                "prag_vlm": prag_vlm,
                "prag_llm": prag_llm,
                "prag_gap": prag_llm - prag_vlm if not (np.isnan(prag_vlm) or np.isnan(prag_llm)) else np.nan,
                "test_p_value": test_result["p_value"] if test_result is not None else np.nan,
                "test_effect_size": test_result["effect_size"] if test_result is not None else np.nan,
                "test_significant": test_result["significant"] if test_result is not None else False,
            }
        )

    # =========================================================================
    # Part 4: Aggregate results and statistics
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 4: Results Summary")
    print("=" * 80)

    df_results = pd.DataFrame(prag_results)

    # Overall comparison (across all tasks)
    valid_mask = ~(np.isnan(df_results["prag_vlm"]) | np.isnan(df_results["prag_llm"]))
    if valid_mask.sum() > 0:
        vlm_prag_all = df_results.loc[valid_mask, "prag_vlm"].to_numpy()
        llm_prag_all = df_results.loc[valid_mask, "prag_llm"].to_numpy()

        overall_test = stats_obj.test_vlm_vs_llm(vlm_prag_all, llm_prag_all)

        print("\nOverall comparison (across tasks):")
        print(f"  VLM PRAG mean: {overall_test['vlm_mean']:.4f}")
        print(f"  LLM PRAG mean: {overall_test['llm_mean']:.4f}")
        print(f"  Gap (LLM - VLM): {overall_test['gap']:.4f}")
        print(f"  Wilcoxon p-value: {overall_test['p_value']:.4f}")
        print(f"  Cohen's d: {overall_test['effect_size']:.4f}")
        print(f"  Significant: {overall_test['significant']}")
    else:
        overall_test = None

    print("\nPer-task results:")
    print(df_results.to_string(index=False))

    # =========================================================================
    # Part 5: Save results
    # =========================================================================
    results_path = prag_root / "prag_comparison_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        results_dict = {
            "per_task": df_results.to_dict(orient="records"),
            "overall": overall_test if overall_test is not None else None,
            "target_layer": target_layer,
            "model": config.model.model_id,
        }
        # Convert numpy/pandas types to JSON-serializable types
        results_dict_serializable = convert_to_json_serializable(results_dict)
        json.dump(results_dict_serializable, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_path}")

    # Save CSV
    csv_path = prag_root / "prag_comparison_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

    print("\n" + "=" * 80)
    print("Experiment completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
