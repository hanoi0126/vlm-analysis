"""Model registry for feature extractors."""

from src.config.schema import ModelConfig
from src.models.base import BaseFeatureExtractor
from src.models.qwen import QwenVLFeatureExtractor

# Model registry mapping
MODEL_REGISTRY: dict[str, type[BaseFeatureExtractor]] = {
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
        available_models: list[str] = list(MODEL_REGISTRY.keys())
        error_msg: str = f"Model '{config.name}' not found in registry. Available models: {available_models}"
        raise ValueError(error_msg)

    extractor_class: type[BaseFeatureExtractor] = MODEL_REGISTRY[config.name]

    # Create extractor with appropriate kwargs
    if config.name == "qwen":
        return extractor_class(  # type: ignore[call-arg]
            model_id=config.model_id,
            device=config.device,
            int8=config.use_int8,
            use_fast_processor=config.use_fast_processor,
            llm_layers=config.llm_layers,
        )
    # Generic fallback
    return extractor_class(  # type: ignore[call-arg]
        model_id=config.model_id,
        device=config.device,
    )


def register_model(name: str, extractor_class: type[BaseFeatureExtractor]) -> None:
    """
    Register a new model extractor.

    Args:
        name: Model name
        extractor_class: Extractor class
    """
    MODEL_REGISTRY[name] = extractor_class
