"""Script for image vs text-only comparison experiment."""

from pathlib import Path
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.schema import Config
from src.models.registry import create_extractor
from src.probing import probe_all_tasks, run_extract_probe_decode
from src.utils import get_experiment_output_dir
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

    # Set up experiment output directory
    experiment_name = "comparison"
    comparison_root = get_experiment_output_dir(config.output.results_root, experiment_name, config.model.model_id)
    print(f"\nExperiment output directory: {comparison_root}")

    # =========================================================================
    # Part 1: Run with images (Image ON)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: Running WITH images (Image ON)")
    print("=" * 80)

    # Override results_root to use new structure for Image ON
    config.output.results_root = comparison_root

    print(f"\nExtracting features for tasks: {config.experiment.tasks}")
    summary_with = run_extract_probe_decode(
        extractor=extractor,
        config=config,
        use_image=True,
        show_progress=True,
        condition_suffix="_imageon",
    )
    print("\nExtraction summary (Image ON):")
    print(summary_with)

    # Run probing if enabled
    probe_with = None
    if config.probe.enabled:
        print("\nRunning probing experiments (Image ON)...")
        # Modify task names to include suffix for probing
        tasks_imageon = [f"{task}_imageon" for task in config.experiment.tasks]
        probe_with = probe_all_tasks(
            results_root=comparison_root,
            tasks=tasks_imageon,
            n_folds=config.probe.n_folds,
            seed=config.probe.seed,
            max_iter=config.probe.max_iter,
            c_value=config.probe.C,
            solver=config.probe.solver,
            verbose=True,
        )
        print("\nProbing summary (Image ON):")
        print(probe_with)
    else:
        print("\n[SKIP] Probing disabled in config (probe.enabled=false)")

    # =========================================================================
    # Part 2: Run without images (Image OFF - text-only)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: Running WITHOUT images (Image OFF - Text-only with description)")
    print("=" * 80)

    # Override results_root to use new structure for Image OFF
    # (results_root is already set to comparison_root)

    print(f"\nExtracting features for tasks: {config.experiment.tasks}")
    summary_without = run_extract_probe_decode(
        extractor=extractor,
        config=config,
        use_image=False,
        show_progress=True,
        condition_suffix="_imageoff",
    )
    print("\nExtraction summary (Image OFF):")
    print(summary_without)

    # Run probing if enabled
    probe_without = None
    if config.probe.enabled:
        print("\nRunning probing experiments (Image OFF)...")
        # Modify task names to include suffix for probing
        tasks_imageoff = [f"{task}_imageoff" for task in config.experiment.tasks]
        probe_without = probe_all_tasks(
            results_root=comparison_root,
            tasks=tasks_imageoff,
            n_folds=config.probe.n_folds,
            seed=config.probe.seed,
            max_iter=config.probe.max_iter,
            c_value=config.probe.C,
            solver=config.probe.solver,
            verbose=True,
        )
        print("\nProbing summary (Image OFF):")
        print(probe_without)
    else:
        print("\n[SKIP] Probing disabled in config (probe.enabled=false)")

    # =========================================================================
    # Part 3: Generate comparison plots
    # =========================================================================
    if config.output.save_plots and config.probe.enabled:
        print("\n" + "=" * 80)
        print("PART 3: Generating comparison plots")
        print("=" * 80)

        plot_comparison(
            results_root=comparison_root,
            tasks=config.experiment.tasks,
            suffix_with_img="_imageon",
            suffix_no_img="_imageoff",
            title_suffix=f"{config.model.model_id} (k={config.probe.n_folds})",
        )
    elif config.output.save_plots and not config.probe.enabled:
        print("\n[SKIP] Plot generation requires probing (probe.enabled=true)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\nExperiment results: {comparison_root}")
    print(f"  Image ON:  {comparison_root}/*_imageon")
    print(f"  Image OFF: {comparison_root}/*_imageoff")
    # Prepare summary data for markdown table
    summary_rows = []

    if config.probe.enabled and probe_with is not None and probe_without is not None:
        print("\nBest layers comparison:")
        print("-" * 80)
        print(f"{'Task':<15} {'Image ON (acc)':<20} {'Image OFF (acc)':<20} {'Diff':<10}")
        print("-" * 80)

        for task in config.experiment.tasks:
            with_row = probe_with[probe_with["task"] == f"{task}_imageon"]
            without_row = probe_without[probe_without["task"] == f"{task}_imageoff"]

            if not with_row.empty and not without_row.empty:
                acc_with = with_row["best_acc"].values[0]
                acc_without = without_row["best_acc"].values[0]
                diff = acc_with - acc_without
                sign = "+" if diff >= 0 else ""
                
                # Get sample counts
                n_with = with_row["n"].values[0] if "n" in with_row.columns else None
                n_without = without_row["n"].values[0] if "n" in without_row.columns else None
                
                # Determine winner
                if diff > 0:
                    winner = "Image"
                    winner_marker = ""
                elif diff < 0:
                    winner = "Text"
                    winner_marker = "✓✓" if abs(diff) > 0.1 else "✓"
                else:
                    winner = "Tie"
                    winner_marker = ""
                
                print(
                    f"{task:<15} {acc_with:.4f} ({with_row['best_layer'].values[0]:<8}) "
                    f"{acc_without:.4f} ({without_row['best_layer'].values[0]:<8}) {sign}{diff:.4f}"
                )
                
                summary_rows.append({
                    "task": task,
                    "with_image": acc_with,
                    "text_only": acc_without,
                    "diff": diff,
                    "winner": winner,
                    "winner_marker": winner_marker,
                    "n_image": n_with if n_with is not None and not (isinstance(n_with, float) and np.isnan(n_with)) else None,
                    "n_text": n_without if n_without is not None and not (isinstance(n_without, float) and np.isnan(n_without)) else None,
                })
    else:
        # Use decode accuracy if probing is disabled
        print("\nDecode accuracy comparison (probing disabled):")
        print("-" * 80)
        print(f"{'Task':<15} {'Image ON (decode)':<20} {'Image OFF (decode)':<20} {'Diff':<10}")
        print("-" * 80)

        for task in config.experiment.tasks:
            with_row = summary_with[summary_with["task"] == task]
            without_row = summary_without[summary_without["task"] == task]

            if not with_row.empty and not without_row.empty:
                decode_with = with_row["decode_acc"].values[0] if "decode_acc" in with_row.columns else None
                decode_without = without_row["decode_acc"].values[0] if "decode_acc" in without_row.columns else None
                
                if decode_with is not None and decode_without is not None and not (np.isnan(decode_with) or np.isnan(decode_without)):
                    diff = decode_with - decode_without
                    sign = "+" if diff >= 0 else ""
                    
                    # Get sample counts
                    n_with = with_row["n"].values[0] if "n" in with_row.columns else None
                    n_without = without_row["n"].values[0] if "n" in without_row.columns else None
                    
                    # Determine winner
                    if diff > 0:
                        winner = "Image"
                        winner_marker = ""
                    elif diff < 0:
                        winner = "Text"
                        winner_marker = "✓✓" if abs(diff) > 0.1 else "✓"
                    else:
                        winner = "Tie"
                        winner_marker = ""
                    
                    print(
                        f"{task:<15} {decode_with:.4f} {'N/A':<8} "
                        f"{decode_without:.4f} {'N/A':<8} {sign}{diff:.4f}"
                    )
                    
                    summary_rows.append({
                        "task": task,
                        "with_image": decode_with,
                        "text_only": decode_without,
                        "diff": diff,
                        "winner": winner,
                        "winner_marker": winner_marker,
                        "n_image": n_with if n_with is not None and not (isinstance(n_with, float) and np.isnan(n_with)) else None,
                        "n_text": n_without if n_without is not None and not (isinstance(n_without, float) and np.isnan(n_without)) else None,
                    })

    # Save markdown summary
    markdown_path = comparison_root / "comparison_summary.md"
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("# Image vs Text-Only Comparison Results\n\n")
        f.write(f"**Model:** {config.model.model_id}\n")
        f.write(f"**Experiment Date:** {Path(comparison_root).name}\n")
        
        if config.probe.enabled:
            f.write(f"**CV Folds:** {config.probe.n_folds}\n")
            f.write("**Method:** Probing accuracy\n\n")
        else:
            f.write("**Method:** Decode accuracy (probing disabled)\n\n")
        
        if summary_rows:
            f.write("## Summary Table\n\n")
            f.write("| Task | With Image | Text-Only | Diff | Winner | n (image) | n (text) |\n")
            f.write("|------|------------|-----------|------|--------|-----------|----------|\n")
            
            # Sort by diff (descending) to show largest differences first
            summary_rows_sorted = sorted(summary_rows, key=lambda x: x["diff"], reverse=True)
            
            for row in summary_rows_sorted:
                task = row["task"]
                with_img = row["with_image"]
                text_only = row["text_only"]
                diff = row["diff"]
                winner = row["winner"]
                winner_marker = row["winner_marker"]
                n_img = row["n_image"] if row["n_image"] is not None else "N/A"
                n_txt = row["n_text"] if row["n_text"] is not None else "N/A"
                
                diff_str = f"{diff:+.3f}" if diff != 0 else "0.000"
                winner_str = f"{winner} {winner_marker}".strip()
                
                f.write(
                    f"| {task} | {with_img:.3f} | {text_only:.3f} | {diff_str} | {winner_str} | {n_img} | {n_txt} |\n"
                )
            
            f.write("\n## Notes\n\n")
            f.write("- **With Image**: Feature extraction")
            if config.probe.enabled:
                f.write(" and probing")
            f.write(" with images\n")
            f.write("- **Text-Only**: Feature extraction")
            if config.probe.enabled:
                f.write(" and probing")
            f.write(" without images (text-only using description)\n")
            f.write("- **Diff**: With Image - Text-Only (positive = Image better, negative = Text better)\n")
            if config.probe.enabled:
                f.write("- **Winner**: Determined by probing accuracy difference\n")
            else:
                f.write("- **Winner**: Determined by decode accuracy difference\n")
            f.write("  - ✓✓: Large difference (>0.1)\n")
            f.write("  - ✓: Small difference (≤0.1)\n")
        else:
            f.write("## No Results\n\n")
            f.write("No comparison data available.\n")
    
    print(f"\nMarkdown summary saved to: {markdown_path}")

    print("\n" + "=" * 80)
    print("Experiment completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
