"""
HuggingFace Hub Uploader

Upload datasets to HuggingFace Hub.
"""

from pathlib import Path
from typing import Optional

try:
    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from distillo.config import HFUploadConfig


class HFUploader:
    """Upload datasets to HuggingFace Hub"""

    def __init__(self, config: HFUploadConfig):
        if not HF_AVAILABLE:
            raise ImportError(
                "datasets and huggingface_hub are required. "
                "Install with: pip install distillo[server]"
            )

        self.config = config
        self.api = HfApi(token=config.token)

    def upload_from_file(self, file_path: str | Path, format: str = "jsonl") -> str:
        """
        Upload dataset from a file

        Args:
            file_path: Path to the dataset file
            format: File format (jsonl, json, parquet)

        Returns:
            URL of the uploaded dataset
        """
        file_path = Path(file_path)

        # Load dataset from file
        if format == "jsonl":
            dataset = load_dataset("json", data_files=str(file_path))["train"]
        elif format == "json":
            dataset = load_dataset("json", data_files=str(file_path))["train"]
        elif format == "parquet":
            dataset = load_dataset("parquet", data_files=str(file_path))["train"]
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Upload to Hub
        dataset.push_to_hub(
            repo_id=self.config.repo_id,
            config_name=self.config.config_name,
            token=self.config.token,
            private=self.config.private,
            commit_message=self.config.commit_message or "Upload dataset from Distillo",
        )

        return f"https://huggingface.co/datasets/{self.config.repo_id}"

    def upload_from_dict(self, data: list[dict], split: str = "train") -> str:
        """
        Upload dataset from a list of dictionaries

        Args:
            data: List of dictionaries
            split: Dataset split name

        Returns:
            URL of the uploaded dataset
        """
        # Create dataset
        dataset = Dataset.from_list(data)

        # Upload to Hub
        dataset.push_to_hub(
            repo_id=self.config.repo_id,
            config_name=self.config.config_name,
            token=self.config.token,
            private=self.config.private,
            split=split,
            commit_message=self.config.commit_message or "Upload dataset from Distillo",
        )

        return f"https://huggingface.co/datasets/{self.config.repo_id}"
