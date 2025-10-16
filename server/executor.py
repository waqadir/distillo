"""
Job Executor

Orchestrates data loading, model inference, and result writing.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from distillo.config import JobConfig, OutputMode
from server.backends import create_backend
from server.sources import load_data_source
from server.uploader import HFUploader
from server.writers import create_writer

logger = logging.getLogger(__name__)


class JobExecutor:
    """
    Execute data generation jobs

    Orchestrates the complete pipeline:
    1. Load data from source
    2. Generate completions using model backend
    3. Apply optional custom processor
    4. Write results to output
    5. Upload to HuggingFace Hub if configured
    """

    def __init__(self, job_config: JobConfig, artifacts_dir: str | Path = ".artifacts"):
        """
        Initialize job executor

        Args:
            job_config: Job configuration
            artifacts_dir: Directory for storing job artifacts
        """
        self.config = job_config
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.backend = None
        self.processor = None

        # Metrics
        self.metrics = {
            "total_records": 0,
            "processed_records": 0,
            "total_tokens": 0,
            "failed_records": 0,
        }

    def set_processor(self, processor: Callable[[List[Dict]], List[Dict]]) -> None:
        """
        Set custom processor function

        Args:
            processor: Function that takes a list of records and returns processed records
        """
        self.processor = processor

    def execute(self, job_id: str) -> Dict[str, Any]:
        """
        Execute the job

        Args:
            job_id: Unique job identifier

        Returns:
            Execution result with metrics and output path

        Raises:
            Exception: If job execution fails
        """
        logger.info(f"Starting job execution: {job_id}")

        # Check if on-policy training mode
        if self.config.generation.on_policy:
            return self._execute_on_policy(job_id)
        else:
            return self._execute_off_policy(job_id)

    def _execute_off_policy(self, job_id: str) -> Dict[str, Any]:
        """
        Execute off-policy data generation

        Args:
            job_id: Unique job identifier

        Returns:
            Execution result
        """
        try:
            # Initialize backend
            logger.info(f"Initializing backend: {self.config.model.provider}")
            self.backend = create_backend(self.config.model)

            # Load data source
            logger.info(f"Loading data source: {self.config.source.dataset}")
            data_source = load_data_source(self.config.source)

            # Prepare output
            job_artifact_dir = self.artifacts_dir / job_id
            job_artifact_dir.mkdir(parents=True, exist_ok=True)

            output_format = self.config.output.format
            artifact_path = job_artifact_dir / f"result.{output_format}"

            writer = create_writer(artifact_path, format=output_format)

            # Process data
            logger.info("Processing data...")
            batch = []
            batch_size = self.config.generation.max_batch_size or 32

            for record in data_source:
                batch.append(record)

                if len(batch) >= batch_size:
                    self._process_batch(batch, writer)
                    batch = []

            # Process remaining records
            if batch:
                self._process_batch(batch, writer)

            # Finalize output
            writer.close()

            logger.info(f"Job completed. Processed {self.metrics['processed_records']} records")

            # Handle output mode
            result = {"job_id": job_id, "artifact_path": str(artifact_path), "metrics": self.metrics}

            if self.config.output.mode == OutputMode.HF_UPLOAD:
                if self.config.output.hf:
                    logger.info("Uploading to HuggingFace Hub...")
                    uploader = HFUploader(self.config.output.hf)
                    hub_url = uploader.upload_from_file(artifact_path, format=output_format)
                    result["hub_url"] = hub_url
                    logger.info(f"Uploaded to: {hub_url}")

            return result

        except Exception as e:
            logger.error(f"Job execution failed: {str(e)}")
            raise

        finally:
            # Cleanup
            if self.backend:
                self.backend.close()

    def _execute_on_policy(self, job_id: str) -> Dict[str, Any]:
        """
        Execute on-policy RL training

        Args:
            job_id: Unique job identifier

        Returns:
            Training result with metrics and checkpoint path
        """
        if not self.config.generation.on_policy_options:
            raise ValueError("on_policy_options must be set for on-policy training")

        try:
            from server.training.on_policy_trainer import OnPolicyTrainer

            logger.info("Starting on-policy RL training")

            # Initialize teacher backend
            from distillo.config import ModelConfig

            teacher_config = ModelConfig(
                provider="openai",  # Default to OpenAI for teacher
                name=self.config.generation.on_policy_options.teacher,
                parameters={
                    "api_key": self.config.generation.on_policy_options.api_key,
                },
            )
            teacher_backend = create_backend(teacher_config)

            # Initialize on-policy trainer
            checkpoint_dir = self.artifacts_dir / job_id / "checkpoints"
            trainer = OnPolicyTrainer(
                student_model_name=self.config.model.name,
                teacher_backend=teacher_backend,
                config=self.config.generation.on_policy_options,
                checkpoint_dir=checkpoint_dir,
            )

            # Load data source
            logger.info(f"Loading training data: {self.config.source.dataset}")
            data_source = load_data_source(self.config.source)

            # Train
            def data_iterator():
                """Generator for training data"""
                for record in data_source:
                    yield record

            # Run training
            metrics = trainer.train_epoch(data_iterator())

            # Save final checkpoint
            final_checkpoint = trainer.save_checkpoint("final")

            # Optionally merge and save full model
            if self.config.output.mode != OutputMode.RETURN:
                output_path = self.artifacts_dir / job_id / "merged_model"
                trainer.finalize_and_save(output_path)
                artifact_path = output_path
            else:
                artifact_path = final_checkpoint

            result = {
                "job_id": job_id,
                "artifact_path": str(artifact_path),
                "checkpoint_path": str(final_checkpoint),
                "metrics": metrics,
                "training_steps": trainer.global_step,
            }

            logger.info(f"On-policy training completed: {trainer.global_step} steps")

            return result

        except Exception as e:
            logger.error(f"On-policy training failed: {str(e)}")
            raise

        finally:
            # Cleanup
            if hasattr(self, "backend") and self.backend:
                self.backend.close()

    def _process_batch(self, batch: List[Dict[str, Any]], writer) -> None:
        """
        Process a batch of records

        Args:
            batch: List of input records
            writer: Output writer
        """
        try:
            # Extract prompts
            prompts = self._extract_prompts(batch)

            # Apply duplications
            duplications = self.config.generation.duplications
            if duplications > 1:
                prompts = prompts * duplications
                batch = batch * duplications

            # Generate completions
            generation_params = self.config.generation.parameters.copy()
            if self.config.generation.seed is not None:
                generation_params["seed"] = self.config.generation.seed

            results = self.backend.generate(prompts, generation_params)

            # Combine with original records
            output_records = []
            for i, result in enumerate(results):
                output_record = batch[i].copy()
                output_record["generated_text"] = result["text"]
                output_record["tokens"] = result["tokens"]
                output_record["finish_reason"] = result["finish_reason"]
                output_records.append(output_record)

            # Apply custom processor if provided
            if self.processor:
                output_records = self.processor(output_records)

            # Write records
            for record in output_records:
                writer.write_record(record)

            # Update metrics
            self.metrics["total_records"] += len(batch)
            self.metrics["processed_records"] += len(output_records)
            self.metrics["total_tokens"] += sum(r["tokens"] for r in results)

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            self.metrics["failed_records"] += len(batch)
            raise

    def _extract_prompts(self, records: List[Dict[str, Any]]) -> List[str]:
        """
        Extract prompts from records

        Args:
            records: List of records

        Returns:
            List of prompt strings
        """
        prompts = []

        for record in records:
            # If field is specified in config, use that field as prompt
            if self.config.source.field:
                if self.config.source.field in record:
                    prompts.append(str(record[self.config.source.field]))
                else:
                    # Try to get nested field
                    prompt = self._get_nested_value(record, self.config.source.field)
                    prompts.append(str(prompt) if prompt is not None else "")
            else:
                # Otherwise, try common prompt fields
                if "prompt" in record:
                    prompts.append(str(record["prompt"]))
                elif "text" in record:
                    prompts.append(str(record["text"]))
                elif "content" in record:
                    prompts.append(str(record["content"]))
                else:
                    # Use entire record as string
                    prompts.append(str(record))

        return prompts

    @staticmethod
    def _get_nested_value(record: Dict[str, Any], path: str) -> Any:
        """Get nested value using dot notation"""
        parts = path.split(".")
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


def execute_job(job_config: JobConfig, job_id: str, processor: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Convenience function to execute a job

    Args:
        job_config: Job configuration
        job_id: Job identifier
        processor: Optional processor function

    Returns:
        Execution result
    """
    executor = JobExecutor(job_config)

    if processor:
        executor.set_processor(processor)

    return executor.execute(job_id)
