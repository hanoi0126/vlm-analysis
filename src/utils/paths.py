"""Path utilities for organizing experiment results."""

from pathlib import Path


def get_model_short_name(model_id: str) -> str:
    """Extract a short name from model ID for directory naming.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")

    Returns:
        Short model name (e.g., "qwen3b")

    Examples:
        >>> get_model_short_name("Qwen/Qwen2.5-VL-3B-Instruct")
        'qwen3b'
        >>> get_model_short_name("meta-llama/Llama-3.2-11B-Vision-Instruct")
        'llama11b'
    """
    model_id_lower = model_id.lower()

    # Extract model family name
    if "qwen" in model_id_lower:
        family = "qwen"
    elif "llama" in model_id_lower:
        family = "llama"
    elif "phi" in model_id_lower:
        family = "phi"
    elif "gemma" in model_id_lower:
        family = "gemma"
    elif "molmo" in model_id_lower:
        family = "molmo"
    else:
        # Use the first part of the model name
        parts = model_id.split("/")
        if len(parts) > 1:
            family = parts[1].split("-")[0].lower()
        else:
            family = parts[0].split("-")[0].lower()

    # Extract size indicator (e.g., 3b, 7b, 11b)
    size = ""
    for part in model_id_lower.replace("-", " ").replace(".", " ").split():
        if "b" in part and any(c.isdigit() for c in part):
            # Extract number + "b"
            num_part = "".join(c for c in part if c.isdigit() or c == ".")
            if num_part:
                size = num_part.replace(".", "") + "b"
                break

    return f"{family}{size}" if size else family


def get_experiment_output_dir(
    results_root: Path | str,
    experiment_name: str,
    model_id: str,
    task: str | None = None,
) -> Path:
    """Get output directory for experiment results.

    Structure: results/{experiment_name}/{model_name}/{task}/

    Args:
        results_root: Root directory for all results
        experiment_name: Name of experiment (e.g., "comparison", "similarity", "cross_condition")
        model_id: HuggingFace model ID
        task: Task name (optional, if None returns experiment/model dir)

    Returns:
        Path to experiment output directory

    Examples:
        >>> get_experiment_output_dir("results", "similarity", "Qwen/Qwen2.5-VL-3B-Instruct", "color")
        Path('results/similarity/qwen3b/color')
    """
    results_root = Path(results_root)
    model_short_name = get_model_short_name(model_id)

    if task is None:
        return results_root / experiment_name / model_short_name
    return results_root / experiment_name / model_short_name / task


def get_feature_dir(
    results_root: Path | str,
    experiment_name: str,
    model_id: str,
    task: str,
    condition: str,
) -> Path:
    """Get directory for extracted features.

    Args:
        results_root: Root directory for all results
        experiment_name: Name of experiment
        model_id: HuggingFace model ID
        task: Task name
        condition: Condition name (e.g., "imageon", "imageoff")

    Returns:
        Path to feature directory
    """
    base_dir = get_experiment_output_dir(results_root, experiment_name, model_id, task)
    return base_dir / condition
