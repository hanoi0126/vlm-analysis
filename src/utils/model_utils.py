"""Utility functions for model configuration."""

from torch import nn


def get_model_architecture_info(model: nn.Module) -> dict[str, int]:
    """
    Extract architecture information from model config.

    Args:
        model: The VLM model

    Returns:
        Dictionary with 'num_layers' and 'num_heads'
    """

    def _fallback_values() -> dict[str, int]:
        """Return fallback values with warning."""
        print("Warning: Could not auto-detect model architecture")
        print("Using default values: 36 layers, 16 heads (Qwen2.5-VL-3B)")
        return {
            "num_layers": 36,
            "num_heads": 16,
        }

    try:
        # Try to access language model config
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            language_model = model.model.language_model
        elif hasattr(model, "language_model"):
            language_model = model.language_model
        else:
            err_msg = "Cannot find language_model in model"
            raise AttributeError(err_msg)  # noqa: TRY301

        # Get config
        if hasattr(language_model, "config"):
            config = language_model.config
        elif hasattr(model, "config"):
            config = model.config
        else:
            err_msg = "Cannot find config in model"
            raise AttributeError(err_msg)  # noqa: TRY301

        # Extract num_layers
        if hasattr(language_model, "layers"):
            num_layers = len(language_model.layers)  # type: ignore[arg-type]
        elif hasattr(config, "num_hidden_layers"):
            num_layers = int(config.num_hidden_layers)  # type: ignore[arg-type]
        else:
            err_msg = "Cannot determine number of layers"
            raise AttributeError(err_msg)  # noqa: TRY301

        # Extract num_heads
        # For Qwen models: config.num_attention_heads
        if hasattr(config, "num_attention_heads"):
            num_heads = int(config.num_attention_heads)  # type: ignore[arg-type]
        elif hasattr(config, "num_heads"):
            num_heads = int(config.num_heads)  # type: ignore[arg-type]
        else:
            err_msg = "Cannot determine number of attention heads"
            raise AttributeError(err_msg)  # noqa: TRY301

        return {
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
        }

    except AttributeError:
        # Fallback to default values with warning
        return _fallback_values()
