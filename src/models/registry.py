"""Model registry for feature extractors."""

from typing import Dict, Type

from ..config.schema import ModelConfig
from .base import BaseFeatureExtractor
from .qwen import QwenVLFeatureExtractor

# Model registry mapping
MODEL_REGISTRY: Dict[str, Type[BaseFeatureExtractor]] = {
    "qwen": QwenVLFeatureExtractor,
    # Add more models here as needed
    # "llava": LlavaFeatureExtractor,
}


def create_extractor(config: ModelConfig) -> BaseFeatureExtractor:
    """
    Create feature extractor from config.

    Args:
        config: Model configuration

    Returns:
        Initialized feature extractor

    Raises:
        ValueError: If model name not found in registry
    """
    if config.name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{config.name}' not found in registry. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    extractor_class = MODEL_REGISTRY[config.name]

    # Create extractor with appropriate kwargs
    if config.name == "qwen":
        return extractor_class(
            model_id=config.model_id,
            device=config.device,
            int8=config.use_int8,
            use_fast_processor=config.use_fast_processor,
            llm_layers=config.llm_layers,
        )
    else:
        # Generic fallback
        return extractor_class(
            model_id=config.model_id,
            device=config.device,
        )


def register_model(name: str, extractor_class: Type[BaseFeatureExtractor]) -> None:
    """
    Register a new model extractor.

    Args:
        name: Model name
        extractor_class: Extractor class
    """
    MODEL_REGISTRY[name] = extractor_class
