"""Script for feature extraction only (without probing)."""
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.schema import Config
from src.models.registry import create_extractor
from src.probing import run_extract_probe_decode


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Extract features only.
    
    Args:
        cfg: Hydra configuration
    """
    config = Config(**OmegaConf.to_container(cfg, resolve=True))
    
    print(f"Loading model: {config.model.model_id}")
    extractor = create_extractor(config.model)
    
    print(f"Extracting features for tasks: {config.experiment.tasks}")
    summary_df = run_extract_probe_decode(
        extractor=extractor,
        config=config,
        show_progress=True,
    )
    
    print("\nExtraction summary:")
    print(summary_df)
    print("\nFeatures saved successfully!")


if __name__ == "__main__":
    main()
