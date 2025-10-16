"""
Output Writers

Writers for various output formats (JSONL, JSON, Parquet, etc.)
"""

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


class Writer:
    """Base class for output writers"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, records: List[Dict[str, Any]]) -> None:
        """Write records to file"""
        raise NotImplementedError

    def write_record(self, record: Dict[str, Any]) -> None:
        """Write a single record (for streaming)"""
        raise NotImplementedError

    def close(self) -> None:
        """Close writer and finalize file"""
        pass


class JSONLWriter(Writer):
    """JSONL (JSON Lines) writer"""

    def __init__(self, path: str | Path):
        super().__init__(path)
        self._file = None

    def write(self, records: List[Dict[str, Any]]) -> None:
        """Write all records at once"""
        with open(self.path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def write_record(self, record: Dict[str, Any]) -> None:
        """Write a single record (streaming mode)"""
        if self._file is None:
            self._file = open(self.path, "w")
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close file"""
        if self._file is not None:
            self._file.close()
            self._file = None


class JSONWriter(Writer):
    """JSON writer"""

    def __init__(self, path: str | Path):
        super().__init__(path)
        self._records = []

    def write(self, records: List[Dict[str, Any]]) -> None:
        """Write all records at once"""
        with open(self.path, "w") as f:
            json.dump(records, f, indent=2)

    def write_record(self, record: Dict[str, Any]) -> None:
        """Accumulate records for final write"""
        self._records.append(record)

    def close(self) -> None:
        """Write accumulated records"""
        if self._records:
            self.write(self._records)
            self._records = []


class ParquetWriter(Writer):
    """Parquet writer"""

    def __init__(self, path: str | Path):
        if not PYARROW_AVAILABLE:
            raise ImportError("pyarrow is required for Parquet format")
        super().__init__(path)
        self._records = []

    def write(self, records: List[Dict[str, Any]]) -> None:
        """Write all records at once"""
        # Convert to PyArrow table
        table = pa.Table.from_pylist(records)
        pq.write_table(table, self.path)

    def write_record(self, record: Dict[str, Any]) -> None:
        """Accumulate records for final write"""
        self._records.append(record)

    def close(self) -> None:
        """Write accumulated records"""
        if self._records:
            self.write(self._records)
            self._records = []


def create_writer(path: str | Path, format: str = "jsonl") -> Writer:
    """
    Factory function to create appropriate writer

    Args:
        path: Output file path
        format: Output format (jsonl, json, parquet)

    Returns:
        Writer instance

    Raises:
        ValueError: If format is not supported
    """
    format = format.lower()

    if format == "jsonl":
        return JSONLWriter(path)
    elif format == "json":
        return JSONWriter(path)
    elif format == "parquet":
        return ParquetWriter(path)
    else:
        raise ValueError(
            f"Unsupported format: {format}. Supported formats: jsonl, json, parquet"
        )
