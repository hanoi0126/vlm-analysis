"""Script for logit analysis experiment (image vs text-only comparison with logit extraction)."""

from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.schema import Config
from src.models.registry import create_extractor
from src.probing import run_extract_probe_decode
from src.utils import get_experiment_output_dir


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

    print("\nNext steps:")
    print("  1. Use visualization scripts to analyze logit distributions")
    print("  2. Compare confidence scores between image ON/OFF conditions")
    print("  3. Identify cases where predictions differ")

    print("\n" + "=" * 80)
    print("Experiment completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
