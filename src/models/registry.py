"""Model registry for feature extractors."""

from src.config.schema import ModelConfig
from src.models.base import BaseFeatureExtractor
from src.models.gemma import GemmaFeatureExtractor
from src.models.intern_vl import InternVLFeatureExtractor
from src.models.llava import LlavaFeatureExtractor
from src.models.qwen import QwenVLFeatureExtractor

# Model registry mapping
MODEL_REGISTRY: dict[str, type[BaseFeatureExtractor]] = {
    "qwen": QwenVLFeatureExtractor,
    "llava": LlavaFeatureExtractor,
    "intern_vl": InternVLFeatureExtractor,
    "gemma": GemmaFeatureExtractor,
}


def _detect_model_type(model_id: str) -> str | None:
    """
    Auto-detect model type from model_id.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Detected model type or None
    """
    model_id_lower = model_id.lower()
    if "internvl" in model_id_lower or "opengvlab" in model_id_lower:
        return "intern_vl"
    if "qwen" in model_id_lower and "vl" in model_id_lower:
        return "qwen"
    if "llava" in model_id_lower:
        return "llava"
    if "gemma" in model_id_lower or "paligemma" in model_id_lower:
        return "gemma"
    return None


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
    # Auto-detect model type if config.name doesn't match model_id
    model_name = config.name
    detected_type = _detect_model_type(config.model_id)

    if detected_type and detected_type != config.name:
        print(f"⚠️  Warning: config.name='{config.name}' but model_id suggests '{detected_type}'")
        print(f"    Auto-correcting to use '{detected_type}' extractor for {config.model_id}")
        model_name = detected_type

    if model_name not in MODEL_REGISTRY:
        available_models: list[str] = list(MODEL_REGISTRY.keys())
        error_msg: str = f"Model '{model_name}' not found in registry. Available models: {available_models}"
        raise ValueError(error_msg)

    extractor_class: type[BaseFeatureExtractor] = MODEL_REGISTRY[model_name]

    # Create extractor with appropriate kwargs
    if model_name == "qwen":
        return extractor_class(  # type: ignore[call-arg]
            model_id=config.model_id,
            device=config.device,
            int8=config.use_int8,
            use_fast_processor=config.use_fast_processor,
            llm_layers=config.llm_layers,
        )
    if model_name == "llava":
        return extractor_class(  # type: ignore[call-arg]
            model_id=config.model_id,
            device=config.device,
            int8=config.use_int8,
            use_fast_processor=config.use_fast_processor,
            llm_layers=config.llm_layers,
        )
    if model_name == "intern_vl":
        return extractor_class(  # type: ignore[call-arg]
            model_id=config.model_id,
            device=config.device,
            int8=config.use_int8,
            use_fast_processor=config.use_fast_processor,
            llm_layers=config.llm_layers,
        )
    if model_name == "gemma":
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
