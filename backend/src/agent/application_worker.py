"""Sequential orchestration for persistent application preparation jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from src.agent.application_queue import ApplicationQueue, JobState


@dataclass(frozen=True)
class ProcessingResult:
    """A portal processor's persisted outcome for one application job."""

    state: JobState
    detail: str
    current_url: Optional[str] = None
    selected_resume: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class ApplicationWorker:
    """Process one queued job at a time and never let one failure block a run."""

    def __init__(self, queue: ApplicationQueue, processor: Callable[[dict], ProcessingResult]):
        self.queue = queue
        self.processor = processor

    def process_next(self, run_id: int) -> Optional[dict]:
        """Claim and process the next queued job, or return ``None`` when exhausted."""
        job = self.queue.get_next_queued_job(run_id)
        if job is None:
            return None

        job = self.queue.transition_job(
            job["id"],
            JobState.OPENING_PORTAL,
            "Opening application portal.",
            current_url=job["current_url"] or job["url"],
        )
        try:
            result = self.processor(job)
        except Exception as error:
            return self.queue.transition_job(
                job["id"],
                JobState.FAILED_RETRYABLE,
                f"Portal preparation failed: {error}",
                current_url=job["current_url"] or job["url"],
            )

        return self.queue.transition_job(
            job["id"],
            result.state,
            result.detail,
            current_url=result.current_url,
            selected_resume=result.selected_resume,
            metadata=result.metadata,
        )

    def process_all_queued(self, run_id: int) -> list[dict]:
        """Run each available job sequentially until no queued work remains."""
        processed_jobs = []
        while job := self.process_next(run_id):
            processed_jobs.append(job)
        return processed_jobs