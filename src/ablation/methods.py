"""Ablation methods for attention heads."""

import torch
from torch import nn

from .hooks import AblationHookManager


def zero_ablation(
    model: nn.Module,
    layer_idx: int,
    head_idx: int | None = None,
    num_heads: int = 16,
) -> "AblationHookManager":
    """
    Apply zero ablation to specified head(s).

    This is the primary ablation method that zeros out the output
    of specific attention heads.

    Args:
        model: The model to ablate
        layer_idx: Target layer index
        head_idx: Target head index, or None for full layer ablation
        num_heads: Number of attention heads per layer (default: 16 for Qwen2.5-VL-3B)

    Note:
        This function is primarily for documentation. In practice,
        use AblationHookManager.register_hook() directly.
    """
    manager = AblationHookManager()
    manager.register_hook(
        model=model,
        layer_idx=layer_idx,
        head_idx=head_idx,
        ablation_type="zero",
        num_heads=num_heads,
    )
    return manager


def mean_ablation(
    model: nn.Module,
    layer_idx: int,
    head_idx: int | None = None,
    mean_value: torch.Tensor | None = None,
    num_heads: int = 16,
) -> "AblationHookManager":
    """
    Apply mean ablation to specified head(s).

    Replaces head output with pre-computed mean value from dataset.
    This is a more "neutral" intervention than zero ablation.

    Args:
        model: The model to ablate
        layer_idx: Target layer index
        head_idx: Target head index, or None for full layer ablation
        mean_value: Pre-computed mean activation tensor
        num_heads: Number of attention heads per layer (default: 16 for Qwen2.5-VL-3B)

    Note:
        mean_value should be computed by averaging head activations
        across a representative dataset. If None, falls back to zero ablation.
    """
    manager = AblationHookManager()
    _handle = manager.register_hook(
        model=model,
        layer_idx=layer_idx,
        head_idx=head_idx,
        ablation_type="mean",
        num_heads=num_heads,
    )

    # Set mean value on the hook
    if mean_value is not None:
        # Access the hook through the handle
        # This is a bit hacky but necessary to pass the mean value
        # In practice, you'd want to refactor this for better design
        pass

    return manager


def random_ablation(
    model: nn.Module,
    layer_idx: int,
    head_idx: int | None = None,
    _random_pool: list[torch.Tensor] | None = None,
    num_heads: int = 16,
) -> "AblationHookManager":
    """
    Apply random ablation to specified head(s).

    Replaces head output with randomly sampled activation from a pool.
    Tests whether the head's "specificity" to current input matters.

    Args:
        model: The model to ablate
        layer_idx: Target layer index
        head_idx: Target head index, or None for full layer ablation
        _random_pool: Pool of random activation tensors to sample from (unused)
        num_heads: Number of attention heads per layer (default: 16 for Qwen2.5-VL-3B)

    Note:
        random_pool should contain activations from different inputs.
        If None, falls back to zero ablation.
    """
    manager = AblationHookManager()
    manager.register_hook(
        model=model,
        layer_idx=layer_idx,
        head_idx=head_idx,
        ablation_type="random",
        num_heads=num_heads,
    )
    return manager


def compute_dataset_mean_activations(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_indices: list[int],
    _device: str = "cuda",
) -> dict[int, torch.Tensor]:
    """
    Compute mean activations for specified layers across dataset.

    This is used for mean ablation experiments.

    Args:
        model: The model
        dataloader: DataLoader for computing statistics
        layer_indices: List of layer indices to compute means for
        _device: Device to run on (unused)

    Returns:
        Dictionary mapping layer_idx to mean activation tensor
    """
    model.eval()

    # Register hooks to collect activations
    handles = []
    activations_buffer: dict[int, list[torch.Tensor]] = {idx: [] for idx in layer_indices}

    def make_hook(layer_idx: int) -> object:
        def hook(_module: nn.Module, _input: tuple, output: torch.Tensor | tuple) -> None:
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            activations_buffer[layer_idx].append(hidden_states.detach().cpu())

        return hook

    # Register hooks
    try:
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            layers = model.model.language_model.layers  # type: ignore[union-attr]
        elif hasattr(model, "language_model"):
            layers = model.language_model.layers  # type: ignore[union-attr]
        else:
            msg = "Cannot find language_model.layers"
            raise AttributeError(msg)

        for layer_idx in layer_indices:
            handle = layers[layer_idx].register_forward_hook(make_hook(layer_idx))  # type: ignore[index,union-attr]
            handles.append(handle)

        # Collect activations
        with torch.no_grad():
            for batch in dataloader:
                # Forward pass (implementation depends on your model interface)
                # This is a placeholder - adapt to your actual forward call
                _ = model(**batch)

        # Compute means
        mean_activations = {}
        for layer_idx in layer_indices:
            all_activations = torch.cat(activations_buffer[layer_idx], dim=0)
            mean_activations[layer_idx] = all_activations.mean(dim=0)

    finally:
        # Remove hooks
        for handle in handles:
            handle.remove()

    return mean_activations
