from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4


@dataclass
class Job:
    id: str
    status: str = "queued"
    stage: str = "Queued"
    overall_pct: float = 0.0
    detail: str = "Waiting to start"
    asset_id: str | None = None
    error: str | None = None
    counters: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()

    def create(self) -> Job:
        job = Job(id=str(uuid4()))
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return asdict(job) if job else None

    def update(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = datetime.now(timezone.utc).isoformat()


jobs = JobStore()
