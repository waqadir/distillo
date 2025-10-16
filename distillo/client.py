"""
Distillo Client

HTTP client for interacting with the Distillo server.
"""

import inspect
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from distillo.config import AppConfig, _deep_merge


class DistilloClient:
    """
    Client for interacting with a Distillo server

    Usage:
        # From config file
        client = DistilloClient.from_config("config.yaml")

        # From AppConfig object
        config = AppConfig.load("config.yaml")
        client = DistilloClient(config=config)

        # With context manager
        with DistilloClient.from_config("config.yaml") as client:
            submission = client.submit_job()
            job_id = submission["job_id"]
            status = client.poll_job(job_id, wait_for_completion=True)
            if status["status"] == "completed":
                client.download_result(job_id, destination="result.jsonl")
    """

    def __init__(
        self,
        config: AppConfig,
        processor: Optional[Callable] = None,
        http_client: Optional[Any] = None,
    ):
        """
        Initialize Distillo client

        Args:
            config: Application configuration
            processor: Optional callable to process records
            http_client: Optional httpx.Client instance (for testing)
        """
        if not HTTPX_AVAILABLE and http_client is None:
            raise ImportError(
                "httpx is required for DistilloClient. "
                "Install with: pip install distillo[client]"
            )

        self.config = config
        self.processor = processor
        self._http_client = http_client
        self._owned_client = http_client is None

    @classmethod
    def from_config(
        cls, path: str | Path, processor: Optional[Callable] = None, **overrides
    ) -> "DistilloClient":
        """
        Create client from configuration file

        Args:
            path: Path to configuration file
            processor: Optional callable to process records
            **overrides: Override configuration values

        Returns:
            DistilloClient instance
        """
        config = AppConfig.load(path, overrides=overrides if overrides else None)
        return cls(config=config, processor=processor)

    @classmethod
    def from_dict(
        cls, config_dict: Dict[str, Any], processor: Optional[Callable] = None
    ) -> "DistilloClient":
        """
        Create client from configuration dictionary

        Args:
            config_dict: Configuration dictionary
            processor: Optional callable to process records

        Returns:
            DistilloClient instance
        """
        config = AppConfig(**config_dict)
        return cls(config=config, processor=processor)

    def __enter__(self) -> "DistilloClient":
        """Context manager entry"""
        self._ensure_client()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit"""
        self.close()

    def close(self) -> None:
        """Close HTTP client if owned by this instance"""
        if self._owned_client and self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def _ensure_client(self) -> Any:
        """Ensure HTTP client is initialized"""
        if self._http_client is None:
            if not HTTPX_AVAILABLE:
                raise ImportError("httpx is required. Install with: pip install distillo[client]")

            self._http_client = httpx.Client(
                base_url=self.config.server.base_url,
                headers=self._build_headers(),
                verify=self.config.server.verify_tls,
                timeout=self.config.server.request_timeout,
            )
        return self._http_client

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers including authentication"""
        headers = {"Content-Type": "application/json"}
        headers.update(self.config.server.headers)

        if self.config.server.api_key:
            headers["Authorization"] = f"Bearer {self.config.server.api_key}"

        return headers

    def submit_job(self, **overrides) -> Dict[str, Any]:
        """
        Submit a job to the server

        Args:
            **overrides: Override job configuration values

        Returns:
            Job submission response with job_id

        Raises:
            httpx.HTTPError: If request fails
        """
        client = self._ensure_client()

        # Build job payload
        job_dict = self.config.job.model_dump(mode="json", exclude_none=True)

        # Apply overrides
        if overrides:
            job_dict = _deep_merge(job_dict, overrides)

        # Serialize processor if provided
        if self.processor is not None:
            processor_config = self._serialize_processor(self.processor)
            job_dict["processor"] = processor_config

        # Submit job
        response = client.post("/v1/jobs", json=job_dict)
        response.raise_for_status()

        return response.json()

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status

        Args:
            job_id: Job identifier

        Returns:
            Job status information

        Raises:
            httpx.HTTPError: If request fails
        """
        client = self._ensure_client()
        response = client.get(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running job

        Args:
            job_id: Job identifier

        Returns:
            Cancellation response

        Raises:
            httpx.HTTPError: If request fails
        """
        client = self._ensure_client()
        response = client.post(f"/v1/jobs/{job_id}/cancel")
        response.raise_for_status()
        return response.json()

    def download_result(
        self, job_id: str, destination: Optional[str | Path] = None
    ) -> bytes | None:
        """
        Download job result

        Args:
            job_id: Job identifier
            destination: Optional path to save result

        Returns:
            Result bytes if no destination specified, None otherwise

        Raises:
            httpx.HTTPError: If request fails
        """
        client = self._ensure_client()
        response = client.get(f"/v1/jobs/{job_id}/result")
        response.raise_for_status()

        content = response.content

        if destination:
            dest_path = Path(destination)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(content)
            return None
        else:
            return content

    def poll_job(
        self,
        job_id: str,
        interval: float = 5.0,
        timeout: Optional[float] = None,
        wait_for_completion: bool = True,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Poll job status until completion or timeout

        Args:
            job_id: Job identifier
            interval: Polling interval in seconds
            timeout: Maximum time to wait in seconds (None for no limit)
            wait_for_completion: Wait until job reaches terminal state
            callback: Optional callback function called on each status update

        Returns:
            Final job status

        Raises:
            TimeoutError: If timeout is reached
            httpx.HTTPError: If request fails
        """
        start_time = time.time()
        terminal_states = {"completed", "failed", "cancelled"}

        while True:
            status = self.get_job_status(job_id)

            if callback:
                callback(status)

            if status["status"] in terminal_states:
                return status

            if not wait_for_completion:
                return status

            if timeout and (time.time() - start_time) >= timeout:
                raise TimeoutError(f"Job polling timed out after {timeout} seconds")

            time.sleep(interval)

    def _serialize_processor(self, processor: Callable) -> Dict[str, Any]:
        """
        Serialize a processor function for remote execution

        Args:
            processor: Callable to serialize

        Returns:
            Processor configuration dictionary
        """
        source = inspect.getsource(processor)
        name = processor.__name__

        return {"name": name, "source": source, "kwargs": {}}


class DistilloError(Exception):
    """Base exception for Distillo errors"""

    pass


class JobError(DistilloError):
    """Exception raised for job execution errors"""

    def __init__(self, job_id: str, message: str):
        self.job_id = job_id
        super().__init__(f"Job {job_id} error: {message}")


class JobTimeoutError(DistilloError):
    """Exception raised when job times out"""

    def __init__(self, job_id: str, timeout: float):
        self.job_id = job_id
        self.timeout = timeout
        super().__init__(f"Job {job_id} timed out after {timeout} seconds")
