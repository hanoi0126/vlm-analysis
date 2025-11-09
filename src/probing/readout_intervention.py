"""Readout intervention experiment: replacing unembedding with probe-based readout."""

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import unified_collate


class ProbeBasedReadout(nn.Module):
    """
    Probe-based readout head that replaces unembedding.

    This class uses probe weights to map hidden states to class logits,
    then projects to vocabulary space.
    """

    def __init__(
        self,
        probe_weights: np.ndarray | torch.Tensor,
        vocab_ids: list[int],
        vocab_size: int,
        device: str = "cpu",
    ) -> None:
        """
        Initialize probe-based readout.

        Args:
            probe_weights: Probe weight matrix [num_classes, hidden_dim]
            vocab_ids: List of vocabulary IDs corresponding to classes [num_classes]
            vocab_size: Full vocabulary size
            device: Device to use
        """
        super().__init__()

        # Convert to torch tensor if needed
        if isinstance(probe_weights, np.ndarray):
            probe_weights = torch.from_numpy(probe_weights).float()
        else:
            probe_weights = probe_weights.float()

        self.vocab_ids = torch.tensor(vocab_ids, dtype=torch.long, device=device)
        self.vocab_size = vocab_size
        self.num_classes = probe_weights.shape[0]
        self.hidden_dim = probe_weights.shape[1]

        # Create linear layer from probe weights
        self.probe = nn.Linear(self.hidden_dim, self.num_classes, bias=False)
        self.probe.weight.data = probe_weights.to(device)

        # Freeze probe weights
        self.probe.weight.requires_grad = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: map hidden states to vocabulary logits.

        Args:
            hidden_states: Hidden states [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]

        Returns:
            Vocabulary logits [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]
        """
        # Handle sequence dimension
        if hidden_states.ndim == 3:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_states_flat = hidden_states.view(-1, hidden_dim)
            class_logits_flat = self.probe(hidden_states_flat)  # [batch*seq, num_classes]
            class_logits = class_logits_flat.view(batch_size, seq_len, self.num_classes)
        else:
            class_logits = self.probe(hidden_states)  # [batch_size, num_classes]

        # Create full vocabulary logits
        if hidden_states.ndim == 3:
            batch_size, seq_len, _ = class_logits.shape
            vocab_logits = torch.full(
                (batch_size, seq_len, self.vocab_size),
                float("-inf"),
                dtype=class_logits.dtype,
                device=class_logits.device,
            )
            # Map class logits to vocabulary IDs
            vocab_logits[:, :, self.vocab_ids] = class_logits
        else:
            batch_size, _ = class_logits.shape
            vocab_logits = torch.full(
                (batch_size, self.vocab_size),
                float("-inf"),
                dtype=class_logits.dtype,
                device=class_logits.device,
            )
            # Map class logits to vocabulary IDs
            vocab_logits[:, self.vocab_ids] = class_logits

        return vocab_logits


def evaluate_with_probe_readout(
    extractor: Any,
    dataset: Any,
    probe_weights: np.ndarray,
    vocab_ids: list[int],
    batch_size: int = 8,
    max_new_tokens: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """
    Evaluate model with probe-based readout replacement.

    This function temporarily replaces the model's lm_head with ProbeBasedReadout,
    evaluates on the dataset, then restores the original lm_head.

    Args:
        extractor: Feature extractor model
        dataset: Dataset instance
        probe_weights: Probe weight matrix [num_classes, hidden_dim]
        vocab_ids: List of vocabulary IDs corresponding to classes
        batch_size: Batch size for evaluation
        max_new_tokens: Maximum new tokens for generation
        device: Device to use

    Returns:
        Dictionary containing evaluation results:
            - accuracy: Decode accuracy
            - predictions: List of predictions
            - labels: List of ground truth labels
    """
    # Get original lm_head
    model = extractor.model
    original_lm_head = None

    # Try to get original lm_head
    if hasattr(model, "get_output_embeddings"):
        original_lm_head = model.get_output_embeddings()  # type: ignore[no-untyped-call]
    elif hasattr(model, "language_model") and hasattr(model.language_model, "get_output_embeddings"):
        original_lm_head = model.language_model.get_output_embeddings()  # type: ignore[no-untyped-call]
    elif hasattr(model, "lm_head"):
        original_lm_head = model.lm_head

    if original_lm_head is None:
        error_msg = "Could not find original lm_head"
        raise ValueError(error_msg)

    # Get vocabulary size
    if hasattr(original_lm_head, "weight"):
        vocab_size = original_lm_head.weight.shape[0]
    else:
        error_msg = "Could not determine vocabulary size"
        raise ValueError(error_msg)

    # Create probe-based readout
    probe_readout = ProbeBasedReadout(probe_weights, vocab_ids, vocab_size, device=device)

    # Replace lm_head
    if hasattr(model, "set_output_embeddings"):
        model.set_output_embeddings(probe_readout)  # type: ignore[no-untyped-call]
    elif hasattr(model, "language_model") and hasattr(model.language_model, "set_output_embeddings"):
        model.language_model.set_output_embeddings(probe_readout)  # type: ignore[no-untyped-call]
    elif hasattr(model, "lm_head"):
        model.lm_head = probe_readout
    else:
        error_msg = "Could not replace lm_head"
        raise ValueError(error_msg)

    # Evaluate
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=unified_collate)

    all_predictions = []
    all_labels = []

    try:
        with torch.no_grad():
            for batch in dl:
                # Prepare inputs
                images = batch.get("images", None)
                texts = batch.get("texts", None)

                # Forward pass with generation
                output = extractor.forward(
                    images=images,
                    texts=texts,
                    use_image=True,
                    decode=True,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

                # Get predictions and labels
                if output.gen_parsed is not None:
                    all_predictions.extend(output.gen_parsed)

                # Get labels from batch
                if "label_cls" in batch:
                    all_labels.extend(batch["label_cls"])
                elif "labels" in batch:
                    # Convert label IDs to class names
                    label_ids = batch["labels"]
                    if hasattr(dataset, "id2cls"):
                        all_labels.extend([dataset.id2cls[int(lid)] for lid in label_ids])
                    else:
                        all_labels.extend([str(int(lid)) for lid in label_ids])

    finally:
        # Restore original lm_head
        if hasattr(model, "set_output_embeddings"):
            model.set_output_embeddings(original_lm_head)  # type: ignore[no-untyped-call]
        elif hasattr(model, "language_model") and hasattr(model.language_model, "set_output_embeddings"):
            model.language_model.set_output_embeddings(original_lm_head)  # type: ignore[no-untyped-call]
        elif hasattr(model, "lm_head"):
            model.lm_head = original_lm_head

    # Calculate accuracy
    if len(all_predictions) != len(all_labels):
        print(f"[WARN] Prediction/label count mismatch: {len(all_predictions)} vs {len(all_labels)}")
        min_len = min(len(all_predictions), len(all_labels))
        all_predictions = all_predictions[:min_len]
        all_labels = all_labels[:min_len]

    correct = [p == label for p, label in zip(all_predictions, all_labels, strict=False)]
    accuracy = float(np.mean(correct)) if correct else 0.0

    return {
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
        "n_samples": len(all_predictions),
    }


def compare_baseline_vs_intervention(
    extractor: Any,
    dataset: Any,
    probe_weights: np.ndarray,
    vocab_ids: list[int],
    baseline_results: dict[str, Any] | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """
    Compare baseline (original unembedding) vs intervention (probe-based readout).

    Args:
        extractor: Feature extractor model
        dataset: Dataset instance
        probe_weights: Probe weight matrix [num_classes, hidden_dim]
        vocab_ids: List of vocabulary IDs corresponding to classes
        baseline_results: Baseline evaluation results (if None, will be computed)
        batch_size: Batch size for evaluation
        max_new_tokens: Maximum new tokens for generation
        device: Device to use

    Returns:
        Dictionary containing comparison results:
            - baseline_acc: Baseline accuracy
            - intervention_acc: Intervention accuracy
            - improvement: Improvement (intervention - baseline)
            - relative_improvement: Relative improvement percentage
    """
    # Get baseline results if not provided
    if baseline_results is None:
        # Evaluate baseline (original unembedding)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=unified_collate)

        baseline_predictions = []
        baseline_labels = []

        with torch.no_grad():
            for batch in dl:
                images = batch.get("images", None)
                texts = batch.get("texts", None)

                output = extractor.forward(
                    images=images,
                    texts=texts,
                    use_image=True,
                    decode=True,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

                if output.gen_parsed is not None:
                    baseline_predictions.extend(output.gen_parsed)

                if "label_cls" in batch:
                    baseline_labels.extend(batch["label_cls"])
                elif "labels" in batch:
                    label_ids = batch["labels"]
                    if hasattr(dataset, "id2cls"):
                        baseline_labels.extend([dataset.id2cls[int(lid)] for lid in label_ids])
                    else:
                        baseline_labels.extend([str(int(lid)) for lid in label_ids])

        baseline_correct = [p == label for p, label in zip(baseline_predictions, baseline_labels, strict=False)]
        baseline_acc = float(np.mean(baseline_correct)) if baseline_correct else 0.0

        baseline_results = {
            "accuracy": baseline_acc,
            "predictions": baseline_predictions,
            "labels": baseline_labels,
        }

    # Evaluate intervention
    intervention_results = evaluate_with_probe_readout(
        extractor=extractor,
        dataset=dataset,
        probe_weights=probe_weights,
        vocab_ids=vocab_ids,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    baseline_acc = baseline_results["accuracy"]
    intervention_acc = intervention_results["accuracy"]
    improvement = intervention_acc - baseline_acc
    relative_improvement = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0.0

    return {
        "baseline_acc": baseline_acc,
        "intervention_acc": intervention_acc,
        "improvement": improvement,
        "relative_improvement": relative_improvement,
        "baseline_results": baseline_results,
        "intervention_results": intervention_results,
    }
