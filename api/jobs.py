# api/jobs.py
"""
Simple in-memory job manager for background GEO tracker execution.
For production, replace with Redis + Celery or similar.
"""
import uuid
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import traceback


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    run_id: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    
    # Progress tracking
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    current_provider: Optional[str] = None
    current_query: Optional[str] = None
    
    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Result
    result: Optional[Any] = None
    error: Optional[str] = None
    
    @property
    def progress_percent(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return round((self.completed_tasks / self.total_tasks) * 100, 2)
    
    @property
    def estimated_remaining_seconds(self) -> Optional[int]:
        if not self.started_at or self.completed_tasks == 0:
            return None
        elapsed = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        rate = self.completed_tasks / elapsed if elapsed > 0 else 0
        remaining = self.total_tasks - self.completed_tasks
        if rate > 0:
            return int(remaining / rate)
        return None


class JobManager:
    """
    Simple thread-based job manager.
    
    Usage:
        manager = JobManager()
        job_id = manager.submit(my_function, arg1, arg2, kwarg1=value)
        status = manager.get_status(job_id)
        result = manager.get_result(job_id)
    """
    
    def __init__(self, max_workers: int = 4):
        self._jobs: Dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
    
    def create_job(self, total_tasks: int = 0) -> Job:
        """Create a new job and return it."""
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, total_tasks=total_tasks)
        with self._lock:
            self._jobs[job_id] = job
        return job
    
    def submit(
        self, 
        func: Callable, 
        *args, 
        job: Optional[Job] = None,
        **kwargs
    ) -> str:
        """
        Submit a function for background execution.
        Returns the job ID.
        """
        if job is None:
            job = self.create_job()
        
        def wrapper():
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            try:
                # Pass the job to the function so it can update progress
                result = func(*args, job=job, **kwargs)
                job.result = result
                job.status = JobStatus.COMPLETED
            except Exception as e:
                job.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                job.status = JobStatus.FAILED
            finally:
                job.completed_at = datetime.now(timezone.utc)
        
        self._executor.submit(wrapper)
        return job.id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._jobs.get(job_id)
    
    def get_status(self, job_id: str) -> Optional[Dict]:
        """Get job status as a dictionary."""
        job = self.get_job(job_id)
        if not job:
            return None
        return {
            "job_id": job.id,
            "run_id": job.run_id,
            "status": job.status.value,
            "total_tasks": job.total_tasks,
            "completed_tasks": job.completed_tasks,
            "failed_tasks": job.failed_tasks,
            "progress_percent": job.progress_percent,
            "current_provider": job.current_provider,
            "current_query": job.current_query,
            "estimated_remaining_seconds": job.estimated_remaining_seconds,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error": job.error,
        }
    
    def cancel(self, job_id: str) -> bool:
        """
        Request cancellation of a job.
        Note: This only marks the job as cancelled; the actual function
        needs to check job.status periodically to stop early.
        """
        job = self.get_job(job_id)
        if not job:
            return False
        if job.status == JobStatus.RUNNING:
            job.status = JobStatus.CANCELLED
            return True
        return False
    
    def cleanup_old_jobs(self, max_age_seconds: int = 3600):
        """Remove jobs older than max_age_seconds."""
        now = datetime.now(timezone.utc)
        with self._lock:
            to_remove = []
            for job_id, job in self._jobs.items():
                if job.completed_at:
                    age = (now - job.completed_at).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(job_id)
            for job_id in to_remove:
                del self._jobs[job_id]
    
    def list_jobs(self, limit: int = 50) -> list:
        """List recent jobs."""
        jobs = sorted(
            self._jobs.values(),
            key=lambda j: j.created_at,
            reverse=True
        )[:limit]
        return [self.get_status(j.id) for j in jobs]


# Global job manager instance
job_manager = JobManager(max_workers=4)
