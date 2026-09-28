"""FastAPI boundary for the review-first application workflow."""

from __future__ import annotations

import json
import os
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent.application_queue import ApplicationQueue, BLOCKED_STATES, JobState
from src.agent.application_worker import ApplicationWorker
from src.agent.portal_processor import ReviewFirstPortalProcessor
from src.profile.application_answers import save_reusable_narrative_answer
from src.utils.runtime_paths import data_path


BACKEND_DIR = Path(__file__).resolve().parent
configured_data_dir = Path(os.getenv("APPLYAGENT_DATA_DIR", "")) if os.getenv("APPLYAGENT_DATA_DIR") else BACKEND_DIR / "data"
if not configured_data_dir.is_absolute():
    configured_data_dir = BACKEND_DIR / configured_data_dir
os.environ["APPLYAGENT_DATA_DIR"] = str(configured_data_dir)

configured_queue_database = Path(os.getenv("APPLICATION_QUEUE_DB", "")) if os.getenv("APPLICATION_QUEUE_DB") else configured_data_dir / "applications" / "application_queue.sqlite3"
if not configured_queue_database.is_absolute():
    configured_queue_database = BACKEND_DIR / configured_queue_database
os.environ["APPLICATION_QUEUE_DB"] = str(configured_queue_database)

PROFILE_PATH = data_path("profile", "profile.json")


class CreateRunRequest(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=100)


class ApproveQuestionRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=10_000)
    scope: Literal["job_only", "profile_reusable"] = "profile_reusable"


def _queue() -> ApplicationQueue:
    return ApplicationQueue()


def _profile() -> dict:
    if not PROFILE_PATH.is_file():
        return {}
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _write_profile(profile: dict) -> dict:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = PROFILE_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(PROFILE_PATH)
    return profile


def _raise_not_found(error: ValueError) -> None:
    raise HTTPException(status_code=404, detail=str(error)) from error


def _prepare_next_job(run_id: int) -> None:
    queue = _queue()
    worker = ApplicationWorker(queue, ReviewFirstPortalProcessor(queue=queue))
    worker.process_next(run_id)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _queue()
    yield


app = FastAPI(
    title="ApplyAgent API",
    version="1.0.0",
    description="Review-first job application preparation. This API never submits an application.",
    lifespan=lifespan,
)

origins = [origin.strip() for origin in os.getenv("API_CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/api/v1/health")
def health() -> dict:
    return {"status": "ok", "service": "applyagent-api", "submission_mode": "review_only"}


@app.get("/api/v1/profile")
def get_profile() -> dict:
    return _profile()


@app.put("/api/v1/profile")
def update_profile(profile: dict) -> dict:
    return _write_profile(profile)


@app.get("/api/v1/dashboard")
def dashboard() -> dict:
    queue = _queue()
    runs = queue.list_runs()
    jobs = [job for run in runs for job in queue.list_jobs(run["id"])]
    states = Counter(job["state"] for job in jobs)
    return {
        "runs": runs,
        "metrics": {
            "total_jobs": len(jobs),
            "queued": states[JobState.QUEUED.value],
            "preparing": sum(states[state.value] for state in {
                JobState.OPENING_PORTAL, JobState.EXTRACTING_JD, JobState.SELECTING_RESUME,
                JobState.UPLOADING_RESUME, JobState.RECONCILING_AUTOFILL, JobState.FILLING_FIELDS,
            }),
            "needs_review": sum(states[state.value] for state in BLOCKED_STATES),
            "ready_for_review": states[JobState.READY_FOR_REVIEW.value],
            "llm_spend_today": queue.daily_llm_cost(),
            "llm_daily_budget": float(os.getenv("LLM_DAILY_BUDGET_USD", "1.00")),
        },
    }


@app.get("/api/v1/runs")
def list_runs() -> list[dict]:
    return _queue().list_runs()


@app.post("/api/v1/runs", status_code=201)
def create_run(request: CreateRunRequest) -> dict:
    try:
        return _queue().create_run(request.urls)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/v1/runs/{run_id}")
def get_run(run_id: int) -> dict:
    queue = _queue()
    try:
        return {"run": queue.get_run(run_id), "jobs": queue.list_jobs(run_id)}
    except ValueError as error:
        _raise_not_found(error)


@app.post("/api/v1/runs/{run_id}/prepare-next", status_code=202)
def prepare_next(run_id: int, background_tasks: BackgroundTasks) -> dict:
    queue = _queue()
    try:
        job = queue.get_next_queued_job(run_id)
        if job is None:
            return {"status": "idle", "message": "No queued application remains in this run."}
        background_tasks.add_task(_prepare_next_job, run_id)
        return {"status": "scheduled", "job_id": job["id"], "message": "Application preparation has started."}
    except ValueError as error:
        _raise_not_found(error)


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: int) -> dict:
    queue = _queue()
    try:
        job = queue._get_job(job_id)
    except ValueError as error:
        _raise_not_found(error)
    return {
        "job": job,
        "events": queue.list_events(job_id),
        "blocked_questions": queue.list_blocked_questions(job_id),
        "llm_calls": queue.list_llm_calls(job_id),
    }


@app.post("/api/v1/jobs/{job_id}/retry")
def retry_job(job_id: int) -> dict:
    try:
        return _queue().retry_job(job_id)
    except ValueError as error:
        _raise_not_found(error)


@app.post("/api/v1/questions/{question_id}/approve")
def approve_question(question_id: int, request: ApproveQuestionRequest) -> dict:
    queue = _queue()
    try:
        question = queue.approve_blocked_question(question_id, request.answer, request.scope)
    except ValueError as error:
        _raise_not_found(error)
    if request.scope == "profile_reusable":
        save_reusable_narrative_answer(PROFILE_PATH, question["field_key"], request.answer)
    return question


@app.post("/api/v1/questions/{question_id}/reject")
def reject_question(question_id: int) -> dict:
    try:
        return _queue().reject_blocked_question(question_id)
    except ValueError as error:
        _raise_not_found(error)