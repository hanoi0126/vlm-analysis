"""Readout intervention experiment: replacing unembedding with probe-based readout."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
        vocab_ids: list[int] | list[list[int]],
        vocab_size: int,
        device: str = "cpu",
    ) -> None:
        """
        Initialize probe-based readout.

        Args:
            probe_weights: Probe weight matrix [num_classes, hidden_dim]
            vocab_ids: List of vocabulary IDs corresponding to classes.
                      Can be list[int] for single-token classes or list[list[int]] for multi-token classes.
                      For multi-token classes, logits will be assigned to all token positions.
            vocab_size: Full vocabulary size
            device: Device to use
        """
        super().__init__()

        # Convert to torch tensor if needed
        if isinstance(probe_weights, np.ndarray):
            probe_weights = torch.from_numpy(probe_weights).float()
        else:
            probe_weights = probe_weights.float()

        # Handle multi-token vocab_ids
        # Convert to list of lists if needed
        if vocab_ids and isinstance(vocab_ids[0], int):
            # Single-token format: [id1, id2, ...]
            self.vocab_ids_list = [[vid] for vid in vocab_ids]
        else:
            # Multi-token format: [[id1a, id1b], [id2], ...]
            self.vocab_ids_list = [list(vids) if isinstance(vids, (list, tuple)) else [vids] for vids in vocab_ids]

        # Flatten for backward compatibility
        self.vocab_ids = torch.tensor([ids[0] for ids in self.vocab_ids_list], dtype=torch.long, device=device)

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
        # Ensure float32 dtype to match probe weights
        hidden_states = hidden_states.float()

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
            # Map class logits to vocabulary IDs (handle multi-token classes)
            for class_idx, token_ids in enumerate(self.vocab_ids_list):
                # Convert token_ids to tensor if needed
                # token_ids is always list[int] from vocab_ids_list
                token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=vocab_logits.device)

                # Assign same logit to all tokens in multi-token class
                # Use individual assignment to avoid indexing issues
                for token_id in token_ids_tensor:
                    # Extract class logit and remove last dimension: [batch, seq, 1] -> [batch, seq]
                    vocab_logits[:, :, token_id] = class_logits[:, :, class_idx]
        else:
            batch_size, _ = class_logits.shape
            vocab_logits = torch.full(
                (batch_size, self.vocab_size),
                float("-inf"),
                dtype=class_logits.dtype,
                device=class_logits.device,
            )
            # Map class logits to vocabulary IDs (handle multi-token classes)
            for class_idx, token_ids in enumerate(self.vocab_ids_list):
                # Convert token_ids to tensor if needed
                # token_ids is always list[int] from vocab_ids_list
                token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=vocab_logits.device)

                # Assign same logit to all tokens in multi-token class
                # Use individual assignment to avoid indexing issues
                for token_id in token_ids_tensor:
                    # Extract class logit: [batch, 1] -> [batch]
                    vocab_logits[:, token_id] = class_logits[:, class_idx]

        return vocab_logits


def evaluate_with_probe_readout(
    extractor: Any,
    dataset: Any,
    probe_weights: np.ndarray,
    vocab_ids: list[int] | list[list[int]] | None = None,
    task_classes: list[str] | None = None,
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
        vocab_ids: List of vocabulary IDs corresponding to classes.
                  Can be list[int] for single-token or list[list[int]] for multi-token.
                  If None, will be inferred from task_classes.
        task_classes: List of class names. Used to infer vocab_ids if vocab_ids is None.
        batch_size: Batch size for evaluation
        max_new_tokens: Maximum new tokens for generation
        device: Device to use

    Returns:
        Dictionary containing evaluation results:
            - accuracy: Decode accuracy
            - predictions: List of predictions
            - labels: List of ground truth labels
    """
    # Infer vocab_ids from task_classes if not provided
    if vocab_ids is None:
        if task_classes is None:
            if hasattr(dataset, "classes"):
                task_classes = dataset.classes
            else:
                error_msg = "Either vocab_ids or task_classes must be provided"
                raise ValueError(error_msg)

        tokenizer = extractor.processor.tokenizer  # type: ignore[union-attr]
        # Get contextual vocab IDs (handle {answer} format)
        # The model outputs answers in {answer} format, so we need to get the token ID
        # for the answer part within the bracketed format
        vocab_ids_list = []
        for cls_name in task_classes:
            # Tokenize answer only (this is what we want to predict)
            answer_only_ids = tokenizer.encode(cls_name, add_special_tokens=False)

            if len(answer_only_ids) == 0:
                # Fallback: use unknown token
                vocab_ids_list.append([0])
            elif len(answer_only_ids) == 1:
                # Single token answer: use it directly
                vocab_ids_list.append([answer_only_ids[0]])
            else:
                # Multi-token answer: use all tokens
                # Note: For multi-token answers, we assign the same logit to all tokens
                vocab_ids_list.append(answer_only_ids)
        vocab_ids = vocab_ids_list
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

    # Debug: print vocab_ids information
    print("[DEBUG] ProbeBasedReadout created:")
    print(f"  num_classes: {probe_readout.num_classes}")
    print(f"  vocab_size: {probe_readout.vocab_size}")
    print(f"  vocab_ids_list: {probe_readout.vocab_ids_list}")
    if task_classes is not None:
        print(f"  task_classes: {task_classes}")
        print("  Mapping:")
        for i, (cls_name, token_ids) in enumerate(zip(task_classes, probe_readout.vocab_ids_list, strict=False)):
            print(f"    Class {i} ('{cls_name}') -> token_ids: {token_ids}")

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
    all_gen_texts = []
    all_questions = []

    try:
        with torch.no_grad():
            for batch in dl:
                # Prepare inputs
                images = batch.get("image", None)  # Note: unified_collate uses "image" not "images"
                texts = batch.get("question", None)  # Use question from batch
                if texts is None:
                    # Fallback: try to get from meta or use empty strings
                    texts = [""] * len(images) if images else [""]

                # Determine use_image based on whether images are available
                # Check if images exist and are not None
                if images is None or (isinstance(images, list) and len(images) == 0):
                    use_image = False
                else:
                    # Check if first image is not None and is a valid PIL Image
                    first_img = images[0] if len(images) > 0 else None
                    use_image = first_img is not None and hasattr(first_img, "size")

                # Forward pass with generation
                output = extractor.forward(
                    images=images if use_image else None,
                    texts=texts,
                    use_image=use_image,
                    decode=True,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

                # Get predictions and labels
                if output.gen_parsed is not None:
                    all_predictions.extend(output.gen_parsed)
                    # Debug: print first few predictions
                    if len(all_predictions) <= 3:
                        print(f"[DEBUG] Sample predictions: {output.gen_parsed[:3]}")
                        print(f"[DEBUG] Sample generated texts: {output.gen_texts[:3] if output.gen_texts else None}")

                # Store generated texts
                if output.gen_texts is not None:
                    all_gen_texts.extend(output.gen_texts)
                else:
                    all_gen_texts.extend([None] * len(texts))

                # Store questions
                all_questions.extend(texts)

                # Get labels from batch
                if "label_cls" in batch:
                    all_labels.extend(batch["label_cls"])
                    # Debug: print first few labels
                    if len(all_labels) <= 3:
                        print(f"[DEBUG] Sample labels: {batch['label_cls'][:3]}")
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

    # Debug: print detailed accuracy information
    if len(all_predictions) > 0 and len(all_labels) > 0:
        print("\n[DEBUG] Intervention evaluation summary:")
        print(f"  Total samples: {len(all_predictions)}")
        print(f"  Correct predictions: {sum(correct)}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  First 10 predictions: {all_predictions[:10]}")
        print(f"  First 10 labels: {all_labels[:10]}")
        print(f"  First 10 correct: {correct[:10]}")

        # Check which predictions are None
        none_count = sum(1 for p in all_predictions if p is None)
        if none_count > 0:
            print(f"  ⚠️  WARNING: {none_count} predictions are None!")

        # Check unique predictions
        unique_preds = set(all_predictions)
        print(f"  Unique predictions: {unique_preds}")

        # Check if predictions are in task vocabulary
        if task_classes is not None:
            task_vocab_set = set(task_classes)
            preds_in_vocab = [p for p in unique_preds if p in task_vocab_set]
            preds_not_in_vocab = [p for p in unique_preds if p not in task_vocab_set]
            print(f"  Predictions in task vocab: {preds_in_vocab}")
            if preds_not_in_vocab:
                print(f"  ⚠️  WARNING: Predictions NOT in task vocab: {preds_not_in_vocab}")

    return {
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
        "gen_texts": all_gen_texts,
        "questions": all_questions,
        "n_samples": len(all_predictions),
    }


def compare_baseline_vs_intervention(
    extractor: Any,
    dataset: Any,
    probe_weights: np.ndarray,
    vocab_ids: list[int] | list[list[int]] | None = None,
    task_classes: list[str] | None = None,
    baseline_results: dict[str, Any] | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 32,
    device: str = "cuda",
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Compare baseline (original unembedding) vs intervention (probe-based readout).

    Args:
        extractor: Feature extractor model
        dataset: Dataset instance
        probe_weights: Probe weight matrix [num_classes, hidden_dim]
        vocab_ids: List of vocabulary IDs corresponding to classes.
                  Can be list[int] for single-token or list[list[int]] for multi-token.
                  If None, will be inferred from task_classes.
        task_classes: List of class names. Used to infer vocab_ids if vocab_ids is None.
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
        baseline_gen_texts = []
        baseline_questions = []

        with torch.no_grad():
            for batch in dl:
                images = batch.get("image", None)  # Note: unified_collate uses "image" not "images"
                texts = batch.get("question", None)  # Use question from batch
                if texts is None:
                    # Fallback: try to get from meta or use empty strings
                    texts = [""] * len(images) if images else [""]

                # Determine use_image based on whether images are available
                # Check if images exist and are not None
                if images is None or (isinstance(images, list) and len(images) == 0):
                    use_image = False
                else:
                    # Check if first image is not None and is a valid PIL Image
                    first_img = images[0] if len(images) > 0 else None
                    use_image = first_img is not None and hasattr(first_img, "size")

                output = extractor.forward(
                    images=images if use_image else None,
                    texts=texts,
                    use_image=use_image,
                    decode=True,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

                if output.gen_parsed is not None:
                    baseline_predictions.extend(output.gen_parsed)

                # Store generated texts
                if output.gen_texts is not None:
                    baseline_gen_texts.extend(output.gen_texts)
                else:
                    baseline_gen_texts.extend([None] * len(texts))

                # Store questions
                baseline_questions.extend(texts)

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
            "gen_texts": baseline_gen_texts,
            "questions": baseline_questions,
        }

    # Evaluate intervention
    intervention_results = evaluate_with_probe_readout(
        extractor=extractor,
        dataset=dataset,
        probe_weights=probe_weights,
        vocab_ids=vocab_ids,
        task_classes=task_classes,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    baseline_acc = baseline_results["accuracy"]
    intervention_acc = intervention_results["accuracy"]
    improvement = intervention_acc - baseline_acc
    relative_improvement = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0.0

    # Prepare data for CSV
    baseline_preds = baseline_results.get("predictions", [])
    intervention_preds = intervention_results.get("predictions", [])
    labels = baseline_results.get("labels", [])
    baseline_gen_texts = baseline_results.get("gen_texts", [])
    intervention_gen_texts = intervention_results.get("gen_texts", [])
    questions = baseline_results.get("questions", []) or intervention_results.get("questions", [])

    # Ensure all lists have the same length
    n_samples = max(
        len(baseline_preds),
        len(intervention_preds),
        len(labels),
        len(baseline_gen_texts),
        len(intervention_gen_texts),
        len(questions),
    )

    # Pad shorter lists with None
    baseline_preds = baseline_preds + [None] * (n_samples - len(baseline_preds))
    intervention_preds = intervention_preds + [None] * (n_samples - len(intervention_preds))
    labels = labels + [None] * (n_samples - len(labels))
    baseline_gen_texts = baseline_gen_texts + [None] * (n_samples - len(baseline_gen_texts))
    intervention_gen_texts = intervention_gen_texts + [None] * (n_samples - len(intervention_gen_texts))
    questions = questions + [None] * (n_samples - len(questions))

    # Create DataFrame
    df_data = {
        "sample_id": list(range(n_samples)),
        "question": questions,
        "label": labels,
        "baseline_prediction": baseline_preds,
        "intervention_prediction": intervention_preds,
        "baseline_correct": [p == label for p, label in zip(baseline_preds, labels, strict=False)],
        "intervention_correct": [p == label for p, label in zip(intervention_preds, labels, strict=False)],
        "baseline_gen_text": baseline_gen_texts,
        "intervention_gen_text": intervention_gen_texts,
    }
    df = pd.DataFrame(df_data)

    # Save CSV if output_path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n[INFO] Intervention results saved to: {output_path}")

    return {
        "baseline_acc": baseline_acc,
        "intervention_acc": intervention_acc,
        "improvement": improvement,
        "relative_improvement": relative_improvement,
        "baseline_results": baseline_results,
        "intervention_results": intervention_results,
        "results_df": df,
    }
