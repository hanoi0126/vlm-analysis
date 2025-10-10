"""Script for image vs text-only comparison experiment."""

from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.schema import Config
from src.models.registry import create_extractor
from src.probing import probe_all_tasks, run_extract_probe_decode
from src.visualization import plot_comparison


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Run image vs text-only comparison experiment.

    This script:
    1. Runs feature extraction + probing WITH images
    2. Runs feature extraction + probing WITHOUT images (text-only using description)
    3. Generates comparison plots

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("Image ON/OFF Comparison Experiment")
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

    # =========================================================================
    # Part 1: Run with images (Image ON)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: Running WITH images (Image ON)")
    print("=" * 80)

    suffix_with_img = config.output.suffix + "_imageon"
    config.output.suffix = suffix_with_img

    print(f"\nExtracting features for tasks: {config.experiment.tasks}")
    summary_with = run_extract_probe_decode(
        extractor=extractor,
        config=config,
        use_image=True,
        show_progress=True,
    )
    print("\nExtraction summary (Image ON):")
    print(summary_with)

    print("\nRunning probing experiments (Image ON)...")
    probe_with = probe_all_tasks(
        results_root=config.output.results_root,
        tasks=config.experiment.tasks,
        suffix=suffix_with_img,
        n_folds=config.probe.n_folds,
        seed=config.probe.seed,
        max_iter=config.probe.max_iter,
        c_value=config.probe.C,
        solver=config.probe.solver,
        verbose=True,
    )
    print("\nProbing summary (Image ON):")
    print(probe_with)

    # =========================================================================
    # Part 2: Run without images (Image OFF - text-only)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: Running WITHOUT images (Image OFF - Text-only with description)")
    print("=" * 80)

    suffix_no_img = config.output.suffix.replace("_imageon", "") + "_imageoff"
    config.output.suffix = suffix_no_img

    print(f"\nExtracting features for tasks: {config.experiment.tasks}")
    summary_without = run_extract_probe_decode(
        extractor=extractor,
        config=config,
        use_image=False,
        show_progress=True,
    )
    print("\nExtraction summary (Image OFF):")
    print(summary_without)

    print("\nRunning probing experiments (Image OFF)...")
    probe_without = probe_all_tasks(
        results_root=config.output.results_root,
        tasks=config.experiment.tasks,
        suffix=suffix_no_img,
        n_folds=config.probe.n_folds,
        seed=config.probe.seed,
        max_iter=config.probe.max_iter,
        c_value=config.probe.C,
        solver=config.probe.solver,
        verbose=True,
    )
    print("\nProbing summary (Image OFF):")
    print(probe_without)

    # =========================================================================
    # Part 3: Generate comparison plots
    # =========================================================================
    if config.output.save_plots:
        print("\n" + "=" * 80)
        print("PART 3: Generating comparison plots")
        print("=" * 80)

        plot_comparison(
            results_root=config.output.results_root,
            tasks=config.experiment.tasks,
            suffix_with_img=suffix_with_img,
            suffix_no_img=suffix_no_img,
            title_suffix=f"{config.model.model_id} (k={config.probe.n_folds})",
        )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\nImage ON results:  {config.output.results_root}/*{suffix_with_img}")
    print(f"Image OFF results: {config.output.results_root}/*{suffix_no_img}")
    print("\nBest layers comparison:")
    print("-" * 80)
    print(f"{'Task':<15} {'Image ON (acc)':<20} {'Image OFF (acc)':<20} {'Diff':<10}")
    print("-" * 80)

    for task in config.experiment.tasks:
        with_row = probe_with[probe_with["task"] == task]
        without_row = probe_without[probe_without["task"] == task]

        if not with_row.empty and not without_row.empty:
            acc_with = with_row["best_acc"].values[0]
            acc_without = without_row["best_acc"].values[0]
            diff = acc_with - acc_without
            sign = "+" if diff >= 0 else ""
            print(
                f"{task:<15} {acc_with:.4f} ({with_row['best_layer'].values[0]:<8}) "
                f"{acc_without:.4f} ({without_row['best_layer'].values[0]:<8}) {sign}{diff:.4f}"
            )

    print("\n" + "=" * 80)
    print("Experiment completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
