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

    dataset_dir: Path = Field(..., description="Dataset directory path")
    csv_filename: str = Field("ALL_metadata.csv", description="CSV filename")


class ExperimentConfig(BaseModel):
    """Experiment configuration."""

    tasks: List[str] = Field(..., description="List of tasks to run")
    task_prompts: Optional[Dict[str, str]] = Field(
        None, description="Prompts for single-obj tasks"
    )


class ProbeConfig(BaseModel):
    """Probing configuration."""

    n_folds: int = Field(5, description="Number of cross-validation folds")
    seed: int = Field(0, description="Random seed")
    max_iter: int = Field(2000, description="Max iterations for LogisticRegression")
    C: float = Field(1.0, description="Inverse regularization strength")
    solver: str = Field("liblinear", description="Solver for LogisticRegression")


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
