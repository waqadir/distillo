"""
Data Sources

Loaders for various data sources (HuggingFace datasets, files, etc.)
"""

from typing import Any, Dict, Iterator, List

try:
    from datasets import load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from distillo.config import SourceConfig, SourceType


class DataSource:
    """Base class for data sources"""

    def __init__(self, config: SourceConfig):
        self.config = config

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over data records"""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return number of records"""
        raise NotImplementedError


class HuggingFaceDataSource(DataSource):
    """HuggingFace dataset source"""

    def __init__(self, config: SourceConfig):
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "datasets library is required. Install with: pip install distillo[server]"
            )
        super().__init__(config)
        self._dataset = None
        self._load_dataset()

    def _load_dataset(self):
        """Load the HuggingFace dataset"""
        load_kwargs = {
            "path": self.config.dataset,
            "split": self.config.split,
            "streaming": self.config.streaming,
        }

        if self.config.config_name:
            load_kwargs["name"] = self.config.config_name

        if self.config.revision:
            load_kwargs["revision"] = self.config.revision

        # Add any additional options
        load_kwargs.update(self.config.options)

        self._dataset = load_dataset(**load_kwargs)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset records"""
        for record in self._dataset:
            # Extract specific field if configured
            if self.config.field:
                if self.config.field in record:
                    yield {self.config.field: record[self.config.field]}
                else:
                    # If field is a nested path (e.g., "messages.0.content")
                    value = self._get_nested_field(record, self.config.field)
                    if value is not None:
                        yield {self.config.field: value}
            else:
                yield record

    def __len__(self) -> int:
        """Return dataset length"""
        if self.config.streaming:
            # Streaming datasets don't have a known length
            return -1
        return len(self._dataset)

    @staticmethod
    def _get_nested_field(record: Dict[str, Any], field_path: str) -> Any:
        """Get nested field value using dot notation"""
        parts = field_path.split(".")
        value = record

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            elif isinstance(value, (list, tuple)):
                try:
                    idx = int(part)
                    value = value[idx]
                except (ValueError, IndexError):
                    return None
            else:
                return None

        return value


def load_data_source(config: SourceConfig) -> DataSource:
    """
    Factory function to create appropriate data source

    Args:
        config: Source configuration

    Returns:
        DataSource instance
    """
    # For now, we only support HuggingFace datasets
    # In the future, could support other sources like local files, APIs, etc.
    return HuggingFaceDataSource(config)
