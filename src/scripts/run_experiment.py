"""Main experiment script using Hydra for configuration."""
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.schema import Config
from src.models.registry import create_extractor
from src.probing import probe_all_tasks, run_extract_probe_decode
from src.visualization import plot_probe_curves_multi


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main experiment function.
    
    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Convert to Pydantic model for validation
    config = Config(**OmegaConf.to_container(cfg, resolve=True))
    
    # Create feature extractor
    print(f"\nLoading model: {config.model.model_id}")
    extractor = create_extractor(config.model)
    print(f"Model loaded successfully")
    print(f"Tap points: {extractor.get_tap_points()[:5]}... ({len(extractor.get_tap_points())} total)")
    
    # Run feature extraction
    print(f"\nExtracting features for tasks: {config.experiment.tasks}")
    summary_df = run_extract_probe_decode(
        extractor=extractor,
        config=config,
        show_progress=True,
    )
    print("\nExtraction summary:")
    print(summary_df)
    
    # Run probing
    print(f"\nRunning probing experiments...")
    probe_df = probe_all_tasks(
        results_root=config.output.results_root,
        tasks=config.experiment.tasks,
        suffix=config.output.suffix,
        n_folds=config.probe.n_folds,
        seed=config.probe.seed,
        max_iter=config.probe.max_iter,
        C=config.probe.C,
        solver=config.probe.solver,
        verbose=True,
    )
    print("\nProbing summary:")
    print(probe_df)
    
    # Plot results if enabled
    if config.output.save_plots:
        print(f"\nGenerating plots...")
        plot_probe_curves_multi(
            results_root=config.output.results_root,
            tasks=config.experiment.tasks,
            suffix=config.output.suffix,
            title_suffix=f"{config.model.model_id} (k={config.probe.n_folds})",
            show_legend=True,
        )
    
    print("\n" + "=" * 80)
    print("Experiment completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
