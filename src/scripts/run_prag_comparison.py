"""Script for VLM vs LLM PRAG comparison experiment."""

import json
from pathlib import Path
import sys
import traceback

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Any

from src.config.schema import Config
from src.data import HuggingFaceDataset
from src.models.registry import create_extractor
from src.probing.prag import extract_probe_weights, get_lm_head_weights, get_task_vocab_embeddings
from src.probing.prag_analysis import analyze_prag_by_attribute, analyze_prag_with_dataset_classes
from src.probing.prag_layers import track_prag_across_layers
from src.probing.prag_statistics import PRAGStatistics
from src.probing.readout_intervention import compare_baseline_vs_intervention
from src.probing.runner import run_extract_probe_decode
from src.utils import get_experiment_output_dir
from src.utils.model_utils import get_model_architecture_info
from src.visualization.prag_plots import plot_prag_figure1, plot_prag_figure2


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

    # Create output directory for data files (plots will be in plots/)
    output_dir = prag_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = prag_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Determine target layer
    target_layer = getattr(config.prag, "target_layer", "last") if hasattr(config, "prag") else "last"
    if target_layer == "last":
        # Use the last layer (same as unembedding layer)
        # Get available tap points and find the last layer
        tap_points = extractor.get_tap_points()
        # Filter layer points (format: l00, l01, ..., l31)
        layer_points = [p for p in tap_points if p.startswith("l") and len(p) >= 2 and p[1:].isdigit()]
        if layer_points:
            # Extract layer numbers and find the maximum
            layer_nums = [int(p[1:]) for p in layer_points]
            last_layer_num = max(layer_nums)
            target_layer = f"l{last_layer_num:02d}"
            print(f"Auto-detected last layer: {target_layer} (from {len(layer_points)} available layers)")
        else:
            # Fallback: try to get from model config
            try:
                arch_info = get_model_architecture_info(extractor.model)  # type: ignore[arg-type]
                num_layers = arch_info["num_layers"]
                last_layer_num = num_layers - 1
                target_layer = f"l{last_layer_num:02d}"
                print(f"Auto-detected last layer from config: {target_layer} ({num_layers} total layers)")
            except Exception:
                # Final fallback: use l19 (middle layer for common models)
                target_layer = "l19"
                print(f"Warning: Could not auto-detect last layer, using default: {target_layer}")
    print(f"Target layer for PRAG: {target_layer}")

    # =========================================================================
    # Part 1: Run with images (VLM condition)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: Running WITH images (VLM condition)")
    print("=" * 80)

    # Set results_root to output_dir so task directories are created in output/
    config.output.results_root = output_dir

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

    # Store data for visualization
    probe_unembedding_data: dict[str, dict[str, np.ndarray]] = {}
    prag_vlm_results_dict: dict[str, dict[str, Any]] = {}
    prag_llm_results_dict: dict[str, dict[str, Any]] = {}

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
        task_dir_vlm = output_dir / f"{task}_vlm"
        task_dir_llm = output_dir / f"{task}_llm"

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
        # Enable debug mode if explicitly enabled in config
        debug_mode = getattr(config.prag, "debug", False)
        if debug_mode:
            print(f"[DEBUG] Debug mode enabled for task: {task}")
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
                debug=debug_mode,
            )
            prag_vlm = prag_vlm_result["prag"]["prag_mean"]
            prag_vlm_results_dict[task] = prag_vlm_result

            # Extract probe and unembedding weights for visualization (use VLM as representative)
            if "task_classes" in prag_vlm_result:
                # Re-extract weights for visualization
                probe_weights_vlm, _ = extract_probe_weights(
                    features_vlm,
                    labels_vlm,
                    max_iter=config.probe.max_iter,
                    C=config.probe.C,
                    solver=config.probe.solver,
                    use_all_data=True,
                )
                lm_head_weights = get_lm_head_weights(extractor)
                tokenizer = extractor.processor.tokenizer  # type: ignore[union-attr]
                # Use multi-token aware embedding extraction
                unembedding_weights_torch, _ = get_task_vocab_embeddings(
                    tokenizer=tokenizer,
                    task_classes=ds_vlm.classes,
                    lm_head_weights=lm_head_weights,
                    use_average_embedding=True,
                    verbose=False,
                )
                unembedding_weights = unembedding_weights_torch.numpy()

                # Align dimensions
                min_classes = min(probe_weights_vlm.shape[0], unembedding_weights.shape[0])
                probe_unembedding_data[task] = {
                    "probe": probe_weights_vlm[:min_classes],
                    "unembedding": unembedding_weights[:min_classes],
                }
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
                debug=debug_mode,
            )
            prag_llm = prag_llm_result["prag"]["prag_mean"]
            prag_llm_results_dict[task] = prag_llm_result
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
    results_path = output_dir / "prag_comparison_results.json"
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
    csv_path = output_dir / "prag_comparison_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

    # =========================================================================
    # Part 6: Generate visualizations
    # =========================================================================
    if config.output.save_plots:
        print("\n" + "=" * 80)
        print("PART 6: Generating visualizations")
        print("=" * 80)

        # Prepare comparison results dict for visualization
        prag_comparison_results = {
            "per_task": df_results.to_dict(orient="records"),
            "overall": overall_test if overall_test is not None else None,
        }

        # Collect layer-wise results for Panel C
        layer_wise_results: dict[str, pd.DataFrame] = {}
        print("\nCollecting layer-wise PRAG data...")
        for task in config.experiment.tasks:
            try:
                # Use VLM condition for layer-wise analysis
                task_dir_vlm = output_dir / f"{task}_vlm"
                if task_dir_vlm.exists():
                    actual_split = task if config.dataset.hf_split == "auto" else config.dataset.hf_split
                    filter_task = None if actual_split == task else task
                    ds_vlm = HuggingFaceDataset(
                        dataset_name=config.dataset.hf_dataset,  # type: ignore[arg-type]
                        split=actual_split,
                        subset=config.dataset.hf_subset,
                        task=filter_task,
                        cache_dir=None,
                    )
                    layer_df = track_prag_across_layers(
                        extractor=extractor,
                        dataset=ds_vlm,
                        results_root=output_dir,
                        task=f"{task}_vlm",
                        layer_names=None,  # Auto-detect
                        max_iter=config.probe.max_iter,
                        C=config.probe.C,
                        solver=config.probe.solver,
                    )
                    if not layer_df.empty:
                        layer_wise_results[task] = layer_df
            except Exception as e:
                print(f"[WARN] Failed to collect layer-wise data for {task}: {e}")

        # Collect attribute analysis for Panel D
        attribute_analysis: pd.DataFrame | None = None
        print("\nCollecting attribute-wise PRAG data...")
        try:
            # Use VLM condition results - need to use task names with _vlm suffix
            vlm_task_names = [f"{task}_vlm" for task in config.experiment.tasks]
            attribute_analysis = analyze_prag_by_attribute(
                extractor=extractor,
                results_root=output_dir,
                attributes=vlm_task_names,
                layer_name=target_layer,
                max_iter=config.probe.max_iter,
                C=config.probe.C,
                solver=config.probe.solver,
            )
            # Rename attributes back to original task names for display
            if attribute_analysis is not None and not attribute_analysis.empty and "attribute" in attribute_analysis.columns:
                attribute_analysis["attribute"] = attribute_analysis["attribute"].str.replace("_vlm", "", regex=False)
        except Exception as e:
            print(f"[WARN] Failed to collect attribute analysis: {e}")

        # Generate Figure 1
        figure1_path = plots_dir / "prag_figure1.png"
        try:
            plot_prag_figure1(
                results_root=output_dir,
                tasks=config.experiment.tasks,
                prag_comparison_results=prag_comparison_results,
                layer_wise_results=layer_wise_results if layer_wise_results else None,
                attribute_analysis=attribute_analysis,
                probe_unembedding_data=probe_unembedding_data if probe_unembedding_data else None,
                target_layer=target_layer,
                output_path=figure1_path,
            )
            print(f"Figure 1 saved to: {figure1_path}")
        except Exception as e:
            print(f"[WARN] Failed to generate Figure 1: {e}")
            traceback.print_exc()

        # =========================================================================
        # Part 7: Run readout intervention experiment and generate Figure 2
        # =========================================================================
        print("\n" + "=" * 80)
        print("PART 7: Running readout intervention experiment")
        print("=" * 80)

        readout_intervention_results: dict[str, Any] | None = None

        # Select a representative task for intervention (use first task or task with lowest PRAG)
        if config.experiment.tasks:
            # Use first task as representative
            intervention_task = config.experiment.tasks[0]
            print(f"\nRunning readout intervention for task: {intervention_task}")

            try:
                # Load dataset and features for intervention task
                actual_split = intervention_task if config.dataset.hf_split == "auto" else config.dataset.hf_split
                filter_task = None if actual_split == intervention_task else intervention_task

                print(f"  Loading dataset: split={actual_split}, task={filter_task}")
                ds_intervention = HuggingFaceDataset(
                    dataset_name=config.dataset.hf_dataset,  # type: ignore[arg-type]
                    split=actual_split,
                    subset=config.dataset.hf_subset,
                    task=filter_task,
                    cache_dir=None,
                )
                print(f"  Dataset loaded: {len(ds_intervention)} samples, classes: {ds_intervention.classes}")

                # Load features and labels for VLM condition
                task_dir_vlm = output_dir / f"{intervention_task}_vlm"
                features_path_vlm = task_dir_vlm / f"features_{target_layer}.npy"
                labels_path_vlm = task_dir_vlm / "labels.npy"

                print("  Checking files:")
                print(f"    Features: {features_path_vlm} (exists: {features_path_vlm.exists()})")
                print(f"    Labels: {labels_path_vlm} (exists: {labels_path_vlm.exists()})")

                if not features_path_vlm.exists():
                    print(f"[ERROR] Features file not found: {features_path_vlm}")
                    print(f"        Expected directory: {task_dir_vlm}")
                if not labels_path_vlm.exists():
                    print(f"[ERROR] Labels file not found: {labels_path_vlm}")

                if features_path_vlm.exists() and labels_path_vlm.exists():
                    print("  Loading features and labels...")
                    features_vlm = np.load(features_path_vlm)
                    labels_vlm = np.load(labels_path_vlm)
                    print(f"  Loaded features: shape={features_vlm.shape}, labels: shape={labels_vlm.shape}")

                    # Extract probe weights
                    print("  Extracting probe weights...")
                    probe_weights, _ = extract_probe_weights(
                        features_vlm,
                        labels_vlm,
                        max_iter=config.probe.max_iter,
                        C=config.probe.C,
                        solver=config.probe.solver,
                        use_all_data=True,
                    )
                    print(f"  Probe weights extracted: shape={probe_weights.shape}")

                    # Run intervention experiment
                    print("  Running baseline vs intervention comparison...")
                    # Set output path for CSV
                    intervention_csv_path = output_dir / f"{intervention_task}_intervention_results.csv"
                    intervention_result = compare_baseline_vs_intervention(
                        extractor=extractor,
                        dataset=ds_intervention,
                        probe_weights=probe_weights,
                        task_classes=ds_intervention.classes,
                        batch_size=getattr(config, "batch_size", 8),
                        max_new_tokens=getattr(config, "max_new_tokens", 32),
                        device=config.model.device,
                        output_path=intervention_csv_path,
                    )

                    readout_intervention_results = {
                        "baseline_acc": intervention_result["baseline_acc"],
                        "intervention_acc": intervention_result["intervention_acc"],
                        "improvement": intervention_result["improvement"],
                        "relative_improvement": intervention_result["relative_improvement"],
                        "task": intervention_task,
                    }

                    print("\n✓ Intervention results:")
                    print(f"  Baseline accuracy: {intervention_result['baseline_acc']:.4f}")
                    print(f"  Intervention accuracy: {intervention_result['intervention_acc']:.4f}")
                    print(f"  Improvement: {intervention_result['improvement']:.4f}")
                    print(f"  Relative improvement: {intervention_result['relative_improvement']:.2f}%")

                else:
                    print(f"[ERROR] Missing required files for intervention task {intervention_task}")
                    print("        Cannot proceed without features and labels files.")

            except Exception as e:
                print(f"[ERROR] Failed to run readout intervention: {e}")
                print(f"        Error type: {type(e).__name__}")
                traceback.print_exc()
                # Set error information for debugging
                readout_intervention_results = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "task": intervention_task if "intervention_task" in locals() else "unknown",
                }
        else:
            print("[WARN] No tasks configured, skipping readout intervention experiment")

        # Generate Figure 2
        figure2_path = plots_dir / "prag_figure2.png"
        try:
            plot_prag_figure2(
                readout_intervention_results=readout_intervention_results,
                output_path=figure2_path,
            )
            print(f"Figure 2 saved to: {figure2_path}")
        except Exception as e:
            print(f"[WARN] Failed to generate Figure 2: {e}")
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Experiment completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
