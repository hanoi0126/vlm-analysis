"""Script for running probing only (features must already exist)."""
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.schema import Config
from src.probing import probe_all_tasks
from src.visualization import plot_probe_curves_multi


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Run probing on existing features.
    
    Args:
        cfg: Hydra configuration
    """
    config = Config(**OmegaConf.to_container(cfg, resolve=True))
    
    print(f"Running probing for tasks: {config.experiment.tasks}")
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
    
    if config.output.save_plots:
        print("\nGenerating plots...")
        plot_probe_curves_multi(
            results_root=config.output.results_root,
            tasks=config.experiment.tasks,
            suffix=config.output.suffix,
            title_suffix=f"{config.model.model_id} (k={config.probe.n_folds})",
        )
    
    print("\nProbing completed!")


if __name__ == "__main__":
    main()
