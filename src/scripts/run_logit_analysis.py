"""Script for logit analysis experiment (image vs text-only comparison with logit extraction)."""

import hydra
from omegaconf import DictConfig, OmegaConf

from src.config.schema import Config
from src.models.registry import create_extractor
from src.probing import run_extract_probe_decode
from src.utils import get_experiment_output_dir
from src.visualization.logit_plots import (
    plot_choice_probabilities_across_layers,
)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Run logit analysis experiment.

    This script:
    1. Runs feature extraction + logit extraction WITH images
    2. Runs feature extraction + logit extraction WITHOUT images (text-only using description)
    3. Saves logits for all layers and all choice options

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("Logit Analysis Experiment")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Convert to Pydantic model for validation
    config = Config(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

    # Enable logit extraction
    if not config.logit_analysis.extract_logits:
        print("\nWARNING: logit_analysis.extract_logits is False. Enabling it for this experiment.")
        config.logit_analysis.extract_logits = True

    # Create feature extractor
    print(f"\nLoading model: {config.model.model_id}")
    extractor = create_extractor(config.model)
    print("Model loaded successfully")
    print(f"Tap points: {extractor.get_tap_points()[:5]}... ({len(extractor.get_tap_points())} total)")

    # Set up experiment output directory
    experiment_name = "logit_analysis"
    logit_root = get_experiment_output_dir(config.output.results_root, experiment_name, config.model.model_id)
    print(f"\nExperiment output directory: {logit_root}")

    # =========================================================================
    # Part 1: Run with images (Image ON)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: Running WITH images (Image ON) - Extracting Logits")
    print("=" * 80)

    # Override results_root to use new structure for Image ON
    config.output.results_root = logit_root

    print(f"\nExtracting features and logits for tasks: {config.experiment.tasks}")
    summary_with = run_extract_probe_decode(
        extractor=extractor,
        config=config,
        use_image=True,
        show_progress=True,
        condition_suffix="_imageon",
    )
    print("\nExtraction summary (Image ON):")
    print(summary_with)

    # =========================================================================
    # Part 2: Run without images (Image OFF - text-only)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: Running WITHOUT images (Image OFF - Text-only with description) - Extracting Logits")
    print("=" * 80)

    print(f"\nExtracting features and logits for tasks: {config.experiment.tasks}")
    summary_without = run_extract_probe_decode(
        extractor=extractor,
        config=config,
        use_image=False,
        show_progress=True,
        condition_suffix="_imageoff",
    )
    print("\nExtraction summary (Image OFF):")
    print(summary_without)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("LOGIT ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nExperiment results: {logit_root}")
    print(f"  Image ON:  {logit_root}/*_imageon")
    print(f"  Image OFF: {logit_root}/*_imageoff")
    print("\nLogits extracted for all layers:")
    print("  - Choice logits: logits_choices_l*.npy")
    print("  - Choice tokens: choice_token_ids.npy")
    print("  - Choice texts: choice_texts.json")
    if config.logit_analysis.save_full_logits:
        print("  - Full vocab logits: logits_all_l*.npy")

    # =========================================================================
    # Part 3: Generate visualizations
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: Generating Visualizations")
    print("=" * 80)

    plots_dir = logit_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect layers from first task
    first_task = config.experiment.tasks[0]
    task_on_dir = logit_root / f"{first_task}_imageon"

    if task_on_dir.exists():
        logit_files = sorted(task_on_dir.glob("logits_choices_l*.npy"))
        layers = [f.stem.replace("logits_choices_", "") for f in logit_files]
        print(f"Detected layers: {layers}")
    else:
        print("Warning: Could not detect layers. Skipping visualization.")
        layers = []

    if layers:
        for task in config.experiment.tasks:
            print(f"\n--- Visualizing task: {task} ---")

            task_on_dir = logit_root / f"{task}_imageon"
            task_off_dir = logit_root / f"{task}_imageoff"

            if not task_on_dir.exists() or not task_off_dir.exists():
                print(f"  Warning: Task directories not found, skipping {task}")
                continue

            # Create task output directory
            task_output = plots_dir / task
            task_output.mkdir(parents=True, exist_ok=True)

            # Visualize for each layer
            # for layer in layers:
            #     print(f"  Layer: {layer}")

            #     # Load data
            #     data_on = load_logit_data(task_on_dir, layer)
            #     data_off = load_logit_data(task_off_dir, layer)

            #     if "choice_logits" not in data_on or "choice_logits" not in data_off:
            #         print(f"    Warning: Logit data not found for {layer}")
            #         continue

            #     try:
            #         # Heatmap
            #         plot_logit_heatmap(
            #             data_on["choice_logits"],
            #             data_off["choice_logits"],
            #             data_on.get("choice_texts", []),
            #             data_on["labels"],
            #             layer,
            #             task_output / f"{task}_{layer}_heatmap.png",
            #             title=task.upper(),
            #         )
            #         print("    ✓ Heatmap saved")

            #         # Scatter plot
            #         plot_logit_scatter(
            #             data_on["choice_logits"],
            #             data_off["choice_logits"],
            #             data_on.get("choice_texts", []),
            #             data_on["labels"],
            #             layer,
            #             task_output / f"{task}_{layer}_scatter.png",
            #             title=task.upper(),
            #         )
            #         print("    ✓ Scatter plot saved")

            #         # Confidence distribution
            #         plot_confidence_distribution(
            #             data_on["choice_logits"],
            #             data_off["choice_logits"],
            #             data_on["labels"],
            #             layer,
            #             task_output / f"{task}_{layer}_confidence.png",
            #             title=task.upper(),
            #         )
            #         print("    ✓ Confidence distribution saved")

            #         # Mismatch analysis
            #         df_mismatch = analyze_mismatch_cases(
            #             logit_root,
            #             task,
            #             layer,
            #             "_imageon",
            #             "_imageoff",
            #             task_output / f"{task}_{layer}_mismatch.csv",
            #         )
            #         if not df_mismatch.empty:
            #             print(f"    ✓ Mismatch analysis saved ({len(df_mismatch)} cases)")

            #     except Exception as e:
            #         print(f"    Error generating plots for {layer}: {e}")

            # # Layer ranking changes (using multiple layers)
            # try:
            #     plot_layer_ranking_changes(
            #         logit_root,
            #         task,
            #         layers,
            #         "_imageon",
            #         "_imageoff",
            #         task_output / f"{task}_layer_ranking.png",
            #         title=task.upper(),
            #     )
            #     print("  ✓ Layer ranking plot saved")
            # except Exception as e:
            #     print(f"  Error generating layer ranking plot: {e}")

            # # Choice probabilities across layers (mean)
            # try:
            #     plot_choice_probabilities_across_layers(
            #         logit_root,
            #         task,
            #         layers,
            #         "_imageon",
            #         "_imageoff",
            #         task_output / f"{task}_choice_probs_mean.png",
            #         use_mean=True,
            #     )
            #     print("  ✓ Choice probabilities (mean) plot saved")
            # except Exception as e:
            #     print(f"  Error generating choice probabilities (mean) plot: {e}")

            # Choice probabilities across layers (samples)
            try:
                plot_choice_probabilities_across_layers(
                    results_root=logit_root,
                    task=task,
                    layers=layers,
                    suffix_with_img="_imageon",
                    suffix_no_img="_imageoff",
                    output_path=task_output / f"{task}_choice_probs_samples.png",
                    num_samples=5,
                    use_mean=False,
                )
                print("  ✓ Choice probabilities (samples) plot saved")
            except Exception as e:
                print(f"  Error generating choice probabilities (samples) plot: {e}")

        print(f"\n✓ All visualizations saved to: {plots_dir}")

    print("\n" + "=" * 80)
    print("Experiment completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
