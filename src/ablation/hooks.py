"""Forward hooks for attention head ablation."""

import torch
from torch import nn


class HeadAblationHook:
    """
    Forward hook for ablating specific attention heads.

    This hook intercepts the output of a transformer decoder layer and
    zeros out the specified attention head(s).

    For Qwen2.5-VL architecture:
    - Qwen2.5-VL-3B: 16 query heads per layer, 2 KV heads
    - Qwen2.5-VL-7B: 28 query heads per layer, 4 KV heads
    - We ablate query heads by zeroing their attention output
    """

    def __init__(
        self,
        layer_idx: int,
        head_idx: int | None = None,
        ablation_type: str = "zero",
        num_heads: int = 16,
    ) -> None:
        """
        Initialize head ablation hook.

        Args:
            layer_idx: Target layer index
            head_idx: Target head index, or None for full layer ablation
            ablation_type: Type of ablation ('zero', 'mean', 'random')
            num_heads: Number of attention heads per layer (default: 16 for Qwen2.5-VL-3B)
        """
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.ablation_type = ablation_type
        self.num_heads = num_heads

        # Statistics for mean/random ablation (to be populated later)
        self.mean_value: torch.Tensor | None = None
        self.random_pool: list[torch.Tensor] = []

    def __call__(
        self,
        _module: nn.Module,
        input_tuple: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """
        Pre-hook function called before o_proj forward pass.

        This hook intercepts the concatenated attention output (before o_proj),
        reshapes it to separate heads, zeros out the target head, and flattens back.

        Args:
            _module: The o_proj module (unused)
            input_tuple: Input tensors to o_proj (concat_output,)

        Returns:
            Modified input tuple with ablated head(s)
        """
        # Extract concatenated attention output from input tuple
        concat_output = input_tuple[0]

        # Skip ablation if not a tensor or has wrong shape
        if not isinstance(concat_output, torch.Tensor):
            return input_tuple  # type: ignore[unreachable]

        if concat_output.dim() != 3:
            print(f"Warning: Expected 3D concat_output but got {concat_output.dim()}D with shape {concat_output.shape}. Skipping ablation.")
            return input_tuple

        # Apply ablation
        try:
            if self.ablation_type == "zero":
                modified_output = self._zero_ablation(concat_output)
            elif self.ablation_type == "mean":
                modified_output = self._mean_ablation(concat_output)
            elif self.ablation_type == "random":
                modified_output = self._random_ablation(concat_output)
            else:
                msg = f"Unknown ablation type: {self.ablation_type}"
                raise ValueError(msg)  # noqa: TRY301
        except Exception as e:
            print(f"Error during ablation: {e}. Returning original input.")
            return input_tuple

        # Return modified input as tuple (required by pre_hook)
        return (modified_output,)

    def _zero_ablation(self, concat_output: torch.Tensor) -> torch.Tensor:
        """
        Zero out specific head(s) by reshaping concat output.

        The concat_output from attention (before o_proj) has shape:
        [batch_size, seq_len, hidden_dim] where hidden_dim = num_heads * head_dim

        We reshape to [batch_size, seq_len, num_heads, head_dim] to isolate heads,
        zero out the target head, then flatten back.
        """
        batch_size, seq_len, hidden_dim = concat_output.shape
        head_dim = hidden_dim // self.num_heads

        # Reshape to separate heads: [batch, seq, hidden] -> [batch, seq, num_heads, head_dim]
        reshaped = concat_output.view(batch_size, seq_len, self.num_heads, head_dim)

        # Clone to avoid in-place modification issues
        reshaped = reshaped.clone()

        if self.head_idx is None:
            # Full layer ablation: zero out all heads
            reshaped[:, :, :, :] = 0.0
        else:
            # Single head ablation: zero out specific head
            reshaped[:, :, self.head_idx, :] = 0.0

        # Flatten back: [batch, seq, num_heads, head_dim] -> [batch, seq, hidden]
        modified_output = reshaped.view(batch_size, seq_len, hidden_dim)

        return modified_output

    def _mean_ablation(self, concat_output: torch.Tensor) -> torch.Tensor:
        """Replace head output with pre-computed mean value."""
        if self.mean_value is None:
            # Fallback to zero if mean not computed
            return self._zero_ablation(concat_output)

        batch_size, seq_len, hidden_dim = concat_output.shape
        head_dim = hidden_dim // self.num_heads

        # Reshape to separate heads
        reshaped = concat_output.view(batch_size, seq_len, self.num_heads, head_dim)
        reshaped = reshaped.clone()

        if self.head_idx is None:
            # Full layer ablation
            reshaped[:, :, :, :] = self.mean_value
        else:
            # Single head ablation
            reshaped[:, :, self.head_idx, :] = self.mean_value

        # Flatten back
        modified_output = reshaped.view(batch_size, seq_len, hidden_dim)
        return modified_output

    def _random_ablation(self, concat_output: torch.Tensor) -> torch.Tensor:
        """Replace head output with random sample from pool."""
        if not self.random_pool:
            # Fallback to zero if pool not populated
            return self._zero_ablation(concat_output)

        batch_size, seq_len, hidden_dim = concat_output.shape
        head_dim = hidden_dim // self.num_heads

        # Reshape to separate heads
        reshaped = concat_output.view(batch_size, seq_len, self.num_heads, head_dim)
        reshaped = reshaped.clone()

        # Sample random value from pool
        random_idx = int(torch.randint(0, len(self.random_pool), (1,)).item())
        random_value = self.random_pool[random_idx]

        if self.head_idx is None:
            # Full layer ablation
            reshaped[:, :, :, :] = random_value
        else:
            # Single head ablation
            reshaped[:, :, self.head_idx, :] = random_value

        # Flatten back
        modified_output = reshaped.view(batch_size, seq_len, hidden_dim)
        return modified_output


class AblationHookManager:
    """
    Manager for registering and removing ablation hooks.

    Provides context manager interface for temporary ablation.
    """

    def __init__(self) -> None:
        """Initialize hook manager."""
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

    def register_hook(
        self,
        model: nn.Module,
        layer_idx: int,
        head_idx: int | None = None,
        ablation_type: str = "zero",
        num_heads: int = 16,
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Register ablation hook on specified layer's o_proj module.

        Args:
            model: The model (e.g., Qwen2.5-VL model)
            layer_idx: Target layer index
            head_idx: Target head index, or None for full layer
            ablation_type: Type of ablation
            num_heads: Number of heads per layer (default: 16 for Qwen2.5-VL-3B)

        Returns:
            Removable handle for the registered hook
        """
        # Get the target layer's o_proj module
        # For Qwen2.5-VL: model.model.language_model.layers[layer_idx].self_attn.o_proj
        try:
            if hasattr(model, "model") and hasattr(model.model, "language_model"):
                # Standard Qwen2.5-VL structure
                layers = model.model.language_model.layers  # type: ignore[union-attr]
            elif hasattr(model, "language_model"):
                # Alternative structure
                layers = model.language_model.layers  # type: ignore[union-attr]
            else:
                msg = "Cannot find language_model.layers in model"
                raise AttributeError(msg)  # noqa: TRY301

            target_layer = layers[layer_idx]  # type: ignore[index]

            # Target the o_proj module specifically (output projection of attention)
            if hasattr(target_layer, "self_attn") and hasattr(target_layer.self_attn, "o_proj"):
                target_module = target_layer.self_attn.o_proj
            else:
                msg = f"Layer {layer_idx} does not have self_attn.o_proj module"
                raise AttributeError(msg)  # noqa: TRY301

        except (AttributeError, IndexError) as e:
            msg = f"Cannot access layer {layer_idx} or its self_attn.o_proj module. Error: {e}"
            raise ValueError(msg) from e

        # Create hook
        hook = HeadAblationHook(
            layer_idx=layer_idx,
            head_idx=head_idx,
            ablation_type=ablation_type,
            num_heads=num_heads,
        )

        # Register pre-forward hook on o_proj to intercept concat output before projection
        handle = target_module.register_forward_pre_hook(hook)
        self.hooks.append(handle)

        return handle

    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def __enter__(self) -> "AblationHookManager":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit - remove all hooks."""
        self.remove_all_hooks()


def ablate_head(
    model: nn.Module,
    layer_idx: int,
    head_idx: int | None = None,
    ablation_type: str = "zero",
    num_heads: int = 28,
) -> AblationHookManager:
    """
    Context manager for temporary head ablation.

    Usage:
        with ablate_head(model, layer_idx=15, head_idx=3):
            output = model(input)

    Args:
        model: The model
        layer_idx: Target layer index
        head_idx: Target head index, or None for full layer
        ablation_type: Type of ablation
        num_heads: Number of heads per layer

    Returns:
        AblationHookManager context manager
    """
    manager = AblationHookManager()
    manager.register_hook(model, layer_idx, head_idx, ablation_type, num_heads)
    return manager
