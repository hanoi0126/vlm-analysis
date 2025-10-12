"""Base class for feature extractors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class TapOutput:
    """Output from feature extraction."""

    v_enc: torch.Tensor | None = None  # (B, D) — Vision Encoder output
    v_proj: torch.Tensor | None = None  # (B, D) — Vision projected to language space
    layers: dict[str, torch.Tensor] = field(default_factory=dict)  # {'l00': (B,D), ...}
    gen_texts: list[str] | None = None  # Generated texts (new tokens only)
    gen_parsed: list[str | None] | None = None  # Parsed {answer}


class BaseFeatureExtractor(ABC, torch.nn.Module):
    """Abstract base class for VLM feature extractors."""

    def __init__(self, device: str = "cuda") -> None:
        """
        Initialize base extractor.

        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def forward(
        self,
        images: list,
        texts: list[str] | None = None,
        *,
        decode: bool = False,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        generation_kwargs: dict | None = None,
    ) -> TapOutput:
        """
        Extract features from images and texts.

        Args:
            images: List of PIL images
            texts: List of text prompts (optional)
            decode: Whether to decode generated text
            max_new_tokens: Maximum new tokens for generation
            do_sample: Use sampling for generation
            generation_kwargs: Additional generation arguments

        Returns:
            TapOutput containing extracted features
        """

    @abstractmethod
    def get_tap_points(self) -> list[str]:
        """
        Get list of available feature tap points.

        Returns:
            List of tap point names (e.g., ['v_enc', 'v_proj', 'l00', ...])
        """
