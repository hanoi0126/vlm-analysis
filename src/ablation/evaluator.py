"""Evaluation framework for ablation experiments."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import HuggingFaceDataset, unified_collate

from .hooks import AblationHookManager
from .statistics import (
    bootstrap_confidence_interval,
    compute_effect_size,
    compute_prediction_entropy,
    compute_probability_margin,
    permutation_test,
)


class AblationEvaluator:
    """
    Evaluation framework for attention head ablation experiments.

    This class provides methods for:
    - Position-dependent accuracy measurement
    - Multiple evaluation metrics
    - Batch processing with progress bars
    - Statistical validation
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: str = "cuda",
        batch_size: int = 8,
        num_heads: int = 16,
        evaluation_position: str = "answer_head",
    ) -> None:
        """
        Initialize ablation evaluator.

        Args:
            model: The VLM model (e.g., Qwen2.5-VL)
            processor: Model processor/tokenizer
            device: Device to run on
            batch_size: Batch size for evaluation
            num_heads: Number of attention heads per layer (default: 16 for Qwen2.5-VL-3B)
            evaluation_position: Position to evaluate at ('answer_head', 'last', 'mean')
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.evaluation_position = evaluation_position

        self.model.eval()

    def find_answer_head_position(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> list[int]:
        """
        Find the answer-head position (location of '{' token).

        Args:
            input_ids: Input token IDs (shape: [batch_size, seq_len])
            attention_mask: Attention mask (shape: [batch_size, seq_len])

        Returns:
            List of position indices for each sample in batch
        """
        tokenizer = self.processor.tokenizer
        brace_token_id = tokenizer.convert_tokens_to_ids("{")

        if brace_token_id is None:
            # Fallback: try to encode directly
            brace_tokens = tokenizer.encode("{", add_special_tokens=False)
            if len(brace_tokens) > 0:
                brace_token_id = brace_tokens[0]
            else:
                # Last resort: use last valid token
                if attention_mask is not None:
                    return [int(mask.sum() - 1) for mask in attention_mask]
                return [input_ids.shape[1] - 1] * input_ids.shape[0]

        positions = []
        for i, ids in enumerate(input_ids):
            # Find last occurrence of '{' token
            matches = (ids == brace_token_id).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                pos = int(matches[-1])
            # Fallback: use last valid token
            elif attention_mask is not None:
                pos = int(attention_mask[i].sum() - 1)
            else:
                pos = len(ids) - 1
            positions.append(pos)

        return positions

    def evaluate_with_ablation(
        self,
        dataset: HuggingFaceDataset,
        layer_idx: int | None = None,
        head_idx: int | None = None,
        ablation_type: str = "zero",
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """
        Evaluate model with ablation on a dataset.

        Args:
            dataset: Dataset to evaluate on
            layer_idx: Layer to ablate (None = no ablation, baseline)
            head_idx: Head to ablate (None = full layer ablation)
            ablation_type: Type of ablation ('zero', 'mean', 'random')
            show_progress: Show progress bar

        Returns:
            Dictionary with evaluation metrics
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=unified_collate,
        )

        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_top3_predictions = []

        # Setup ablation hook if needed
        hook_manager = None
        if layer_idx is not None:
            hook_manager = AblationHookManager()
            hook_manager.register_hook(
                model=self.model,
                layer_idx=layer_idx,
                head_idx=head_idx,
                ablation_type=ablation_type,
                num_heads=self.num_heads,
            )

        try:
            with torch.no_grad():
                for batch in tqdm(
                    dataloader,
                    desc=f"Eval L{layer_idx}H{head_idx}" if layer_idx is not None else "Baseline",
                    disable=not show_progress,
                ):
                    # Prepare inputs
                    images = batch["image"]
                    questions = batch["question"]
                    answers = batch["answer"]
                    options = batch.get("options", None)

                    # Build chat messages
                    msgs = [
                        [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": img},
                                    {"type": "text", "text": q},
                                ],
                            }
                        ]
                        for img, q in zip(images, questions, strict=False)
                    ]

                    # Apply chat template
                    templated = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]

                    # Tokenize
                    inputs = self.processor(
                        text=templated,
                        images=images,
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Forward pass
                    outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                    logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

                    # Find answer-head positions
                    if self.evaluation_position == "answer_head":
                        positions = self.find_answer_head_position(
                            inputs["input_ids"],
                            inputs.get("attention_mask"),
                        )
                    elif self.evaluation_position == "last":
                        positions = [
                            int(mask.sum() - 1) if mask is not None else logits.shape[1] - 1
                            for mask in inputs.get("attention_mask", [None] * logits.shape[0])
                        ]
                    else:  # mean
                        positions = None

                    # Extract logits at target positions
                    if positions is not None:
                        # Extract logits at specific positions
                        batch_indices = torch.arange(logits.shape[0], device=logits.device)
                        position_tensor = torch.tensor(positions, device=logits.device)
                        target_logits = logits[batch_indices, position_tensor, :]
                    else:
                        # Use mean across sequence
                        target_logits = logits.mean(dim=1)

                    # Get probabilities
                    probs = F.softmax(target_logits, dim=-1)

                    # Get predictions for each choice
                    batch_predictions = []
                    batch_probabilities = []

                    for i, _answer in enumerate(answers):
                        if options is not None and i < len(options):
                            # Multi-choice: compute probability for each option
                            option_probs_list: list[float] = []
                            for option in options[i]:
                                # Tokenize option
                                option_tokens = self.processor.tokenizer.encode(str(option), add_special_tokens=False)
                                if len(option_tokens) == 0:
                                    option_probs_list.append(0.0)
                                    continue

                                # Use first token probability (approximation)
                                token_id = option_tokens[0]
                                option_prob = probs[i, token_id].item()
                                option_probs_list.append(option_prob)

                            # Normalize probabilities
                            option_probs_array = np.array(option_probs_list)
                            if option_probs_array.sum() > 0:
                                option_probs_array = option_probs_array / option_probs_array.sum()

                            # Predict option with highest probability
                            pred_idx = np.argmax(option_probs_array)
                            prediction = options[i][pred_idx]
                            batch_probabilities.append(option_probs_array)
                        else:
                            # Direct prediction from logits
                            pred_token_id = target_logits[i].argmax().item()
                            prediction = self.processor.tokenizer.decode([pred_token_id])
                            batch_probabilities.append(probs[i].cpu().numpy())

                        batch_predictions.append(str(prediction).strip())

                    # Store results
                    all_predictions.extend(batch_predictions)
                    all_labels.extend([str(a).strip() for a in answers])
                    all_probabilities.extend(batch_probabilities)

                    # Get top-3 predictions
                    top3_indices = torch.topk(target_logits, k=3, dim=-1).indices
                    for i in range(len(batch_predictions)):
                        top3_tokens = [self.processor.tokenizer.decode([idx.item()]) for idx in top3_indices[i]]
                        all_top3_predictions.append(top3_tokens)

        finally:
            # Remove hook
            if hook_manager is not None:
                hook_manager.remove_all_hooks()

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)

        # Compute metrics
        accuracy = np.mean(predictions == labels)

        # Top-3 accuracy
        top3_correct = np.array([label in top3_preds for label, top3_preds in zip(labels, all_top3_predictions, strict=False)])
        top3_accuracy = np.mean(top3_correct)

        # Compute entropy and margin (for samples with probability distributions)
        valid_probs = [p for p in all_probabilities if isinstance(p, np.ndarray) and len(p) > 1]
        if valid_probs:
            valid_probs_array = np.array(valid_probs)
            mean_entropy = np.mean(compute_prediction_entropy(valid_probs_array))
            mean_margin = np.mean(compute_probability_margin(valid_probs_array))
        else:
            mean_entropy = np.nan
            mean_margin = np.nan

        results = {
            "accuracy": float(accuracy),
            "top3_accuracy": float(top3_accuracy),
            "mean_entropy": float(mean_entropy),
            "mean_margin": float(mean_margin),
            "n_samples": len(predictions),
            "predictions": predictions,
            "labels": labels,
            "probabilities": all_probabilities,
        }

        return results

    def evaluate_task(
        self,
        dataset_name: str,
        task: str,
        split: str = "train",
        layer_idx: int | None = None,
        head_idx: int | None = None,
        ablation_type: str = "zero",
        show_progress: bool = True,
        cache_dir: str | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate on a specific task.

        Args:
            dataset_name: HuggingFace dataset name
            task: Task name (e.g., 'color', 'angle')
            split: Dataset split
            layer_idx: Layer to ablate (None = baseline)
            head_idx: Head to ablate (None = full layer)
            ablation_type: Type of ablation
            show_progress: Show progress bar
            cache_dir: Cache directory for dataset

        Returns:
            Evaluation results dictionary
        """
        # Load dataset
        try:
            dataset = HuggingFaceDataset(
                dataset_name=dataset_name,
                split=split if split != "auto" else task,
                task=None if split == task else task,
                cache_dir=cache_dir,
            )
        except Exception:
            # Fallback: try with train split and filter by task
            dataset = HuggingFaceDataset(
                dataset_name=dataset_name,
                split="train",
                task=task,
                cache_dir=cache_dir,
            )

        # Evaluate
        results = self.evaluate_with_ablation(
            dataset=dataset,
            layer_idx=layer_idx,
            head_idx=head_idx,
            ablation_type=ablation_type,
            show_progress=show_progress,
        )

        results["task"] = task
        results["layer_idx"] = layer_idx
        results["head_idx"] = head_idx
        results["ablation_type"] = ablation_type

        return results

    def compare_with_baseline(
        self,
        baseline_results: dict[str, Any],
        ablated_results: dict[str, Any],
        compute_statistics: bool = True,
        n_bootstrap: int = 1000,
        n_permutations: int = 1000,
    ) -> dict[str, Any]:
        """
        Compare ablated results with baseline and compute statistics.

        Args:
            baseline_results: Results from baseline evaluation
            ablated_results: Results from ablated evaluation
            compute_statistics: Whether to compute statistical tests
            n_bootstrap: Number of bootstrap samples
            n_permutations: Number of permutations for test

        Returns:
            Comparison results with statistical metrics
        """
        baseline_preds = baseline_results["predictions"]
        baseline_labels = baseline_results["labels"]
        ablated_preds = ablated_results["predictions"]
        ablated_labels = ablated_results["labels"]

        # Compute delta metrics
        delta_acc = ablated_results["accuracy"] - baseline_results["accuracy"]
        delta_top3 = ablated_results["top3_accuracy"] - baseline_results["top3_accuracy"]

        comparison = {
            "baseline_acc": baseline_results["accuracy"],
            "ablated_acc": ablated_results["accuracy"],
            "delta_acc": delta_acc,
            "baseline_top3": baseline_results["top3_accuracy"],
            "ablated_top3": ablated_results["top3_accuracy"],
            "delta_top3": delta_top3,
            "n_samples": len(baseline_labels),
        }

        if compute_statistics:
            # Bootstrap CI for ablated accuracy
            _, ci_lower, ci_upper = bootstrap_confidence_interval(
                ablated_preds,
                ablated_labels,
                n_samples=n_bootstrap,
                confidence=0.95,
            )
            comparison["ablated_ci_lower"] = ci_lower
            comparison["ablated_ci_upper"] = ci_upper

            # Permutation test
            p_value = permutation_test(
                baseline_preds,
                ablated_preds,
                baseline_labels,
                n_permutations=n_permutations,
            )
            comparison["p_value"] = p_value

            # Effect size
            effect_size = compute_effect_size(
                baseline_results["accuracy"],
                ablated_results["accuracy"],
                (baseline_preds == baseline_labels).astype(float),
                (ablated_preds == ablated_labels).astype(float),
            )
            comparison["effect_size"] = effect_size

            # Significance determination
            comparison["is_significant"] = (p_value < 0.001) and (abs(delta_acc) > 0.30)

        return comparison
