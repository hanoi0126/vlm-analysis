"""Configuration schemas using Pydantic BaseModel for Hydra integration."""

from pathlib import Path

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = Field(default=..., description="Model name (e.g., 'qwen')")
    model_id: str = Field(default=..., description="HuggingFace model ID")
    use_int8: bool = Field(default=False, description="Use 8-bit quantization")
    use_fast_processor: bool = Field(default=True, description="Use fast processor")
    llm_layers: str | list[int] = Field(default="all", description="LLM layers to tap")
    device: str = Field(default="cuda", description="Device to use")


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    # Local dataset (CSV-based)
    dataset_dir: Path | None = Field(default=None, description="Dataset directory path (for local CSV)")
    csv_filename: str = Field(default="ALL_metadata.csv", description="CSV filename")

    # HuggingFace dataset
    hf_dataset: str | None = Field(default=None, description="HuggingFace dataset name (e.g., 'username/dataset-name')")
    hf_split: str = Field(default="train", description="HuggingFace dataset split")
    hf_subset: str | None = Field(default=None, description="HuggingFace dataset subset/config name")

    # Image column names for HF datasets
    image_column: str = Field(default="image", description="Image column name in HF dataset")
    label_column: str = Field(default="label", description="Label column name in HF dataset")
    task_column: str = Field(default="task", description="Task column name in HF dataset")


class ExperimentConfig(BaseModel):
    """Experiment configuration."""

    tasks: list[str] = Field(default=..., description="List of tasks to run")
    task_prompts: dict[str, str] | None = Field(
        default=None,
        description="[Deprecated] Prompts for single-obj tasks (now uses 'question' field from dataset)",
    )


class ProbeConfig(BaseModel):
    """Probing configuration."""

    n_folds: int = Field(default=5, description="Number of cross-validation folds")
    seed: int = Field(default=0, description="Random seed")
    max_iter: int = Field(default=2000, description="Max iterations for LogisticRegression")
    C: float = Field(default=1.0, description="Inverse regularization strength")
    solver: str = Field(default="lbfgs", description="Solver for LogisticRegression")


class OutputConfig(BaseModel):
    """Output configuration."""

    results_root: Path = Field(default=..., description="Results root directory")
    save_features: bool = Field(default=True, description="Save feature arrays")
    save_plots: bool = Field(default=True, description="Save plots")


class Config(BaseModel):
    """Main configuration."""

    model: ModelConfig
    dataset: DatasetConfig
    experiment: ExperimentConfig
    probe: ProbeConfig
    output: OutputConfig

    # Experiment settings
    batch_size: int = Field(default=8, description="Batch size")
    device: str = Field(default="cuda", description="Device")
    seed: int = Field(default=0, description="Random seed")

    # Feature extraction settings
    decode: bool = Field(default=True, description="Decode generated text")
    max_new_tokens: int = Field(default=32, description="Max new tokens for generation")
    do_sample: bool = Field(default=False, description="Use sampling for generation")
