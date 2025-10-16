"""
Distillo Server

FastAPI application for job management and execution.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Security
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from distillo.config import JobConfig
from server.executor import JobExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Distillo Server",
    description="Lightweight on/off-policy distillation framework",
    version="0.1.0",
)

# Security
security = HTTPBearer(auto_error=False)

# In-memory job store
_JOB_STORE: Dict[str, "JobRecord"] = {}

# Configuration
ARTIFACTS_DIR = Path(".artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# API key (in production, load from environment or config)
API_KEY = None  # None means no authentication required


class JobRecord(BaseModel):
    """Job record for tracking job state"""

    job_id: str
    status: str = Field(
        ...,
        description="Job status: queued, running, completed, failed, cancelled",
    )
    config: JobConfig
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    artifact_path: Optional[str] = None
    message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class JobSubmission(BaseModel):
    """Job submission response"""

    job_id: str
    status: str


class JobStatus(BaseModel):
    """Job status response"""

    job_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


# Middleware and authentication


async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    """Verify API key if configured"""
    if API_KEY is None:
        return  # No authentication required

    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing API key")

    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# API Routes


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Distillo Server",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/v1/jobs", response_model=JobSubmission)
async def submit_job(
    job_config: JobConfig,
    background_tasks: BackgroundTasks,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    """
    Submit a new job

    Args:
        job_config: Job configuration
        background_tasks: FastAPI background tasks
        credentials: Optional API key credentials

    Returns:
        Job submission response with job_id
    """
    await verify_api_key(credentials)

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Create job record
    job_record = JobRecord(
        job_id=job_id,
        status="queued",
        config=job_config,
        created_at=datetime.utcnow().isoformat(),
    )

    # Store job
    _JOB_STORE[job_id] = job_record

    # Queue background task
    background_tasks.add_task(_execute_job, job_id)

    logger.info(f"Job submitted: {job_id}")

    return JobSubmission(job_id=job_id, status="queued")


@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    """
    Get job status

    Args:
        job_id: Job identifier
        credentials: Optional API key credentials

    Returns:
        Job status information
    """
    await verify_api_key(credentials)

    if job_id not in _JOB_STORE:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job_record = _JOB_STORE[job_id]

    return JobStatus(
        job_id=job_record.job_id,
        status=job_record.status,
        created_at=job_record.created_at,
        started_at=job_record.started_at,
        completed_at=job_record.completed_at,
        message=job_record.message,
        metrics=job_record.metrics,
    )


@app.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    """
    Cancel a job

    Args:
        job_id: Job identifier
        credentials: Optional API key credentials

    Returns:
        Cancellation response
    """
    await verify_api_key(credentials)

    if job_id not in _JOB_STORE:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job_record = _JOB_STORE[job_id]

    # Only cancel if job is queued or running
    if job_record.status in ["queued", "running"]:
        job_record.status = "cancelled"
        job_record.completed_at = datetime.utcnow().isoformat()
        job_record.message = "Job cancelled by user"
        logger.info(f"Job cancelled: {job_id}")
        return {"job_id": job_id, "status": "cancelled"}
    else:
        return {
            "job_id": job_id,
            "status": job_record.status,
            "message": "Job cannot be cancelled in current state",
        }


@app.get("/v1/jobs/{job_id}/result")
async def get_job_result(
    job_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
):
    """
    Get job result

    Args:
        job_id: Job identifier
        credentials: Optional API key credentials

    Returns:
        Job result file or JSON response
    """
    await verify_api_key(credentials)

    if job_id not in _JOB_STORE:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job_record = _JOB_STORE[job_id]

    if job_record.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job_record.status}",
        )

    if not job_record.artifact_path:
        raise HTTPException(status_code=404, detail="No artifact found for job")

    artifact_path = Path(job_record.artifact_path)
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found")

    # Return file
    return FileResponse(
        path=artifact_path,
        filename=f"distillo_{job_id}.{job_record.config.output.format}",
        media_type="application/octet-stream",
    )


# Background task execution


async def _execute_job(job_id: str):
    """
    Execute job in background

    Args:
        job_id: Job identifier
    """
    job_record = _JOB_STORE[job_id]

    try:
        # Update status
        job_record.status = "running"
        job_record.started_at = datetime.utcnow().isoformat()
        logger.info(f"Job started: {job_id}")

        # Load processor if configured
        processor = None
        if job_record.config.processor:
            processor = _load_processor(job_record.config.processor.source, job_record.config.processor.name)

        # Execute job
        executor = JobExecutor(job_record.config, artifacts_dir=ARTIFACTS_DIR)
        if processor:
            executor.set_processor(processor)

        result = executor.execute(job_id)

        # Update status
        job_record.status = "completed"
        job_record.completed_at = datetime.utcnow().isoformat()
        job_record.artifact_path = result["artifact_path"]
        job_record.metrics = result["metrics"]
        job_record.message = "Job completed successfully"

        logger.info(f"Job completed: {job_id}")

    except Exception as e:
        # Update status on failure
        job_record.status = "failed"
        job_record.completed_at = datetime.utcnow().isoformat()
        job_record.message = str(e)
        logger.error(f"Job failed: {job_id} - {str(e)}")


def _load_processor(source_code: str, func_name: str):
    """
    Load processor function from source code

    Args:
        source_code: Python source code
        func_name: Function name to extract

    Returns:
        Processor function
    """
    import types

    # Create a module to hold the function
    module = types.ModuleType("processor_module")

    # Execute source code in module namespace
    exec(source_code, module.__dict__)

    # Get function
    if not hasattr(module, func_name):
        raise ValueError(f"Processor function '{func_name}' not found in source code")

    return getattr(module, func_name)


# Exception handlers


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000, workers=1)
