"""Configuration schemas using Pydantic BaseModel for Hydra integration."""

from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = Field(..., description="Model name (e.g., 'qwen')")
    model_id: str = Field(..., description="HuggingFace model ID")
    use_int8: bool = Field(False, description="Use 8-bit quantization")
    use_fast_processor: bool = Field(True, description="Use fast processor")
    llm_layers: Union[str, List[int]] = Field("all", description="LLM layers to tap")
    device: str = Field("cuda", description="Device to use")


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    # Local dataset (CSV-based)
    dataset_dir: Optional[Path] = Field(
        None, description="Dataset directory path (for local CSV)"
    )
    csv_filename: str = Field("ALL_metadata.csv", description="CSV filename")

    # HuggingFace dataset
    hf_dataset: Optional[str] = Field(
        None, description="HuggingFace dataset name (e.g., 'username/dataset-name')"
    )
    hf_split: str = Field("train", description="HuggingFace dataset split")
    hf_subset: Optional[str] = Field(
        None, description="HuggingFace dataset subset/config name"
    )

    # Image column names for HF datasets
    image_column: str = Field("image", description="Image column name in HF dataset")
    label_column: str = Field("label", description="Label column name in HF dataset")
    task_column: str = Field("task", description="Task column name in HF dataset")


class ExperimentConfig(BaseModel):
    """Experiment configuration."""

    tasks: List[str] = Field(..., description="List of tasks to run")
    task_prompts: Optional[Dict[str, str]] = Field(
        None,
        description="[Deprecated] Prompts for single-obj tasks (now uses 'question' field from dataset)",
    )


class ProbeConfig(BaseModel):
    """Probing configuration."""

    n_folds: int = Field(5, description="Number of cross-validation folds")
    seed: int = Field(0, description="Random seed")
    max_iter: int = Field(2000, description="Max iterations for LogisticRegression")
    C: float = Field(1.0, description="Inverse regularization strength")
    solver: str = Field("lbfgs", description="Solver for LogisticRegression")


class OutputConfig(BaseModel):
    """Output configuration."""

    results_root: Path = Field(..., description="Results root directory")
    save_features: bool = Field(True, description="Save feature arrays")
    save_plots: bool = Field(True, description="Save plots")
    suffix: str = Field("_qwen3b_llmtap", description="Suffix for output directories")


class Config(BaseModel):
    """Main configuration."""

    model: ModelConfig
    dataset: DatasetConfig
    experiment: ExperimentConfig
    probe: ProbeConfig
    output: OutputConfig

    # Experiment settings
    batch_size: int = Field(8, description="Batch size")
    device: str = Field("cuda", description="Device")
    seed: int = Field(0, description="Random seed")

    # Feature extraction settings
    decode: bool = Field(True, description="Decode generated text")
    max_new_tokens: int = Field(32, description="Max new tokens for generation")
    do_sample: bool = Field(False, description="Use sampling for generation")
