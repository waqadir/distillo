"""
Distillo Configuration System

Defines all configuration models using Pydantic for validation and serialization.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    """Data source type"""

    HF_DATASET = "hf_dataset"
    REMOTE_URI = "remote_uri"


class SourceConfig(BaseModel):
    """Configuration for data source loading"""

    dataset: str = Field(..., description="Dataset identifier (e.g., HuggingFace repo ID)")
    config_name: Optional[str] = Field(None, description="Dataset configuration name")
    split: str = Field("train", description="Dataset split to use")
    field: Optional[str] = Field(None, description="Specific field to extract from records")
    revision: Optional[str] = Field(None, description="Dataset revision/version")
    streaming: bool = Field(False, description="Use streaming mode instead of downloading")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional load options")


class ModelConfig(BaseModel):
    """Configuration for model backend"""

    provider: str = Field(..., description="Model provider (e.g., 'vllm', 'openai', 'anthropic')")
    name: str = Field(..., description="Model name or identifier")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional model parameters"
    )


class OnPolicyConfig(BaseModel):
    """Configuration for on-policy training"""

    teacher: str = Field(..., description="Teacher model identifier")
    api_key: Optional[str] = Field(None, description="API key for teacher model service")
    learning_rate: float = Field(1e-4, description="Learning rate for training")
    groups_per_batch: int = Field(512, description="Number of groups per batch")
    group_size: int = Field(4, description="Size of each group")
    max_tokens: int = Field(4096, description="Maximum tokens per generation")
    lora_rank: int = Field(32, description="LoRA rank for efficient fine-tuning")
    loss_fn: Literal["importance_sampling", "ppo"] = Field(
        "importance_sampling", description="Loss function to use"
    )
    kl_penalty_coef: float = Field(0.1, description="KL divergence penalty coefficient")
    compute_post_kl: bool = Field(True, description="Compute post-training KL divergence")
    eval_every: Optional[int] = Field(None, description="Evaluate every N steps")
    save_every: Optional[int] = Field(None, description="Save checkpoint every N steps")


class GenerationConfig(BaseModel):
    """Configuration for data generation process"""

    duplications: int = Field(1, ge=1, description="Number of times to duplicate each prompt")
    max_batch_size: Optional[int] = Field(None, description="Maximum batch size for generation")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Generation parameters (temperature, top_p, etc.)"
    )
    on_policy: bool = Field(False, description="Enable on-policy training mode")
    on_policy_options: Optional[OnPolicyConfig] = Field(
        None, description="On-policy training configuration"
    )

    @field_validator("on_policy_options")
    @classmethod
    def validate_on_policy_options(
        cls, v: Optional[OnPolicyConfig], info
    ) -> Optional[OnPolicyConfig]:
        """Validate that on_policy_options is set when on_policy is True"""
        if info.data.get("on_policy") and v is None:
            raise ValueError("on_policy_options must be set when on_policy is True")
        return v


class OutputMode(str, Enum):
    """Output mode for generated data"""

    RETURN = "return"
    HF_UPLOAD = "upload_hf"
    LOCAL_FILE = "local_file"


class HFUploadConfig(BaseModel):
    """Configuration for HuggingFace Hub upload"""

    repo_id: str = Field(..., description="HuggingFace repository ID")
    config_name: Optional[str] = Field(None, description="Dataset configuration name")
    token: Optional[str] = Field(None, description="HuggingFace access token")
    private: bool = Field(True, description="Make repository private")
    commit_message: Optional[str] = Field(None, description="Custom commit message")


class OutputConfig(BaseModel):
    """Configuration for output handling"""

    mode: OutputMode = Field(OutputMode.RETURN, description="Output mode")
    local_path: Optional[str] = Field(None, description="Local file path for output")
    hf: Optional[HFUploadConfig] = Field(None, description="HuggingFace upload configuration")
    format: str = Field("jsonl", description="Output format (jsonl, json, parquet)")

    @field_validator("hf")
    @classmethod
    def validate_hf_config(cls, v: Optional[HFUploadConfig], info) -> Optional[HFUploadConfig]:
        """Validate that hf is set when mode is HF_UPLOAD"""
        if info.data.get("mode") == OutputMode.HF_UPLOAD and v is None:
            raise ValueError("hf configuration must be set when mode is 'upload_hf'")
        return v


class ProcessorConfig(BaseModel):
    """Configuration for custom data processor"""

    name: str = Field(..., description="Processor function name")
    source: str = Field(..., description="Python source code for processor")
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments for processor"
    )


class JobConfig(BaseModel):
    """Main job configuration"""

    model: ModelConfig = Field(..., description="Model configuration")
    source: SourceConfig = Field(..., description="Data source configuration")
    generation: GenerationConfig = Field(..., description="Generation configuration")
    output: OutputConfig = Field(..., description="Output configuration")
    processor: Optional[ProcessorConfig] = Field(None, description="Custom processor configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Job metadata")


class ServerConfig(BaseModel):
    """Configuration for remote server connection"""

    base_url: str = Field(..., description="Server base URL")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    verify_tls: bool = Field(True, description="Verify TLS certificates")
    request_timeout: float = Field(30.0, description="Request timeout in seconds")
    headers: Dict[str, str] = Field(default_factory=dict, description="Custom HTTP headers")


class AppConfig(BaseModel):
    """Root application configuration"""

    server: ServerConfig = Field(..., description="Server configuration")
    job: JobConfig = Field(..., description="Job configuration")

    @classmethod
    def load(cls, path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> "AppConfig":
        """
        Load configuration from a YAML or JSON file with optional overrides

        Args:
            path: Path to configuration file
            overrides: Optional dictionary to override loaded values

        Returns:
            AppConfig instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            if path.suffix in [".yaml", ".yml"]:
                config_dict = yaml.safe_load(f)
            elif path.suffix == ".json":
                import json

                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")

        # Apply overrides if provided
        if overrides:
            config_dict = _deep_merge(config_dict, overrides)

        return cls(**config_dict)

    def save(self, path: str | Path) -> None:
        """
        Save configuration to a YAML file

        Args:
            path: Path where to save the configuration
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json", exclude_none=True), f, sort_keys=False)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries

    Args:
        base: Base dictionary
        override: Dictionary with override values

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
