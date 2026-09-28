"""Persistent job-application queue and deterministic resume routing."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse


class JobState(str, Enum):
    QUEUED = "queued"
    OPENING_PORTAL = "opening_portal"
    EXTRACTING_JD = "extracting_jd"
    SELECTING_RESUME = "selecting_resume"
    UPLOADING_RESUME = "uploading_resume"
    RECONCILING_AUTOFILL = "reconciling_autofill"
    FILLING_FIELDS = "filling_fields"
    WAITING_LOGIN = "waiting_login"
    WAITING_SIGNUP = "waiting_signup"
    WAITING_VERIFICATION = "waiting_verification"
    WAITING_CAPTCHA = "waiting_captcha"
    WAITING_ANSWER = "waiting_answer"
    WAITING_REVIEW = "waiting_review"
    READY_FOR_REVIEW = "ready_for_review"
    FAILED_RETRYABLE = "failed_retryable"
    SKIPPED = "skipped"


BLOCKED_STATES = {
    JobState.WAITING_LOGIN,
    JobState.WAITING_SIGNUP,
    JobState.WAITING_VERIFICATION,
    JobState.WAITING_CAPTCHA,
    JobState.WAITING_ANSWER,
    JobState.WAITING_REVIEW,
    JobState.FAILED_RETRYABLE,
}

ACTIVE_STATES = {
    JobState.OPENING_PORTAL,
    JobState.EXTRACTING_JD,
    JobState.SELECTING_RESUME,
    JobState.UPLOADING_RESUME,
    JobState.RECONCILING_AUTOFILL,
    JobState.FILLING_FIELDS,
}

TERMINAL_STATES = {JobState.READY_FOR_REVIEW, JobState.SKIPPED}


@dataclass(frozen=True)
class ResumeDecision:
    variant_key: str
    reason: str


def select_resume_variant(job_title: str, job_description: str) -> ResumeDecision:
    """Choose the approved full-stack resume only for explicit frontend work."""
    job_text = f"{job_title} {job_description}".lower()
    frontend_signals = ("full stack", "full-stack", "frontend", "front-end", "react", "typescript", "javascript ui")
    matched_signals = [signal for signal in frontend_signals if signal in job_text]
    if matched_signals:
        return ResumeDecision(
            variant_key="full_stack",
            reason=f"Selected full-stack resume because the JD mentions {', '.join(matched_signals)}.",
        )
    return ResumeDecision(
        variant_key="backend",
        reason="Selected backend resume because the JD has no explicit full-stack or frontend requirement.",
    )


class ApplicationQueue:
    """Durable storage for sequential, review-first application preparation."""

    def __init__(self, database_path: str | Path | None = None):
        self.database_path = Path(database_path or os.getenv("APPLICATION_QUEUE_DB", "data/applications/application_queue.sqlite3"))
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def create_run(self, application_urls: Iterable[str]) -> dict:
        urls = self._unique_urls(application_urls)
        if not urls:
            raise ValueError("At least one valid HTTP(S) application URL is required")

        now = self._timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO application_runs (created_at, updated_at, status) VALUES (?, ?, ?)",
                (now, now, "queued"),
            )
            run_id = cursor.lastrowid
            for position, url in enumerate(urls):
                connection.execute(
                    """INSERT INTO application_jobs
                       (run_id, url, domain, position, state, checkpoint, retry_count, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (run_id, url, urlparse(url).netloc, position, JobState.QUEUED.value, JobState.QUEUED.value, 0, now, now),
                )
        return self.get_run(run_id)

    def get_run(self, run_id: int) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM application_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"Run {run_id} does not exist")
        return dict(row)

    def list_runs(self, limit: int = 50) -> list[dict]:
        """Return recent application runs for operational dashboards."""
        if limit < 1:
            raise ValueError("Run limit must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM application_runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def list_jobs(self, run_id: int) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM application_jobs WHERE run_id = ? ORDER BY position, id", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_next_queued_job(self, run_id: int) -> Optional[dict]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM application_jobs WHERE run_id = ? AND state = ? ORDER BY position, id LIMIT 1",
                (run_id, JobState.QUEUED.value),
            ).fetchone()
        return dict(row) if row else None

    def transition_job(
        self,
        job_id: int,
        next_state: JobState,
        detail: str = "",
        *,
        checkpoint: Optional[JobState] = None,
        current_url: Optional[str] = None,
        selected_resume: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        job = self._get_job(job_id)
        current_state = JobState(job["state"])
        if next_state not in self._allowed_next_states(current_state):
            raise ValueError(f"cannot transition from {current_state.value} to {next_state.value}")

        now = self._timestamp()
        saved_checkpoint = (checkpoint or next_state).value
        with self._connect() as connection:
            connection.execute(
                """UPDATE application_jobs
                   SET state = ?, checkpoint = ?, current_url = COALESCE(?, current_url),
                       selected_resume = COALESCE(?, selected_resume), updated_at = ?
                   WHERE id = ?""",
                (next_state.value, saved_checkpoint, current_url, selected_resume, now, job_id),
            )
            connection.execute(
                """INSERT INTO application_events (job_id, from_state, to_state, detail, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (job_id, current_state.value, next_state.value, detail, json.dumps(metadata or {}), now),
            )
            self._refresh_run_status(connection, job["run_id"], now)
        return self._get_job(job_id)

    def retry_job(self, job_id: int) -> dict:
        job = self._get_job(job_id)
        previous_state = JobState(job["state"])
        if previous_state not in BLOCKED_STATES:
            raise ValueError(f"Job {job_id} is not retryable from {previous_state.value}")

        now = self._timestamp()
        with self._connect() as connection:
            connection.execute(
                """UPDATE application_jobs
                   SET state = ?, retry_count = retry_count + 1, updated_at = ?
                   WHERE id = ?""",
                (JobState.QUEUED.value, now, job_id),
            )
            connection.execute(
                """INSERT INTO application_events (job_id, from_state, to_state, detail, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (job_id, previous_state.value, JobState.QUEUED.value, "Human requested retry", "{}", now),
            )
            self._refresh_run_status(connection, job["run_id"], now)
        return self._get_job(job_id)

    def list_events(self, job_id: int) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM application_events WHERE job_id = ? ORDER BY id", (job_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def record_blocked_questions(self, job_id: int, questions: Iterable[dict]) -> list[dict]:
        """Upsert portal questions that need an explicit human decision."""
        now = self._timestamp()
        with self._connect() as connection:
            for question in questions:
                field_key = str(question["field_key"])
                field_id = str(question.get("field_id", ""))
                connection.execute(
                    """DELETE FROM blocked_questions
                       WHERE job_id = ? AND field_id = ? AND field_key <> ? AND status = 'pending'""",
                    (job_id, field_id, field_key),
                )
                connection.execute(
                    """INSERT INTO blocked_questions
                       (job_id, field_key, field_id, label, field_type, reason, suggested_answer,
                        source, status, scope, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', '', ?, ?)
                       ON CONFLICT(job_id, field_key) DO UPDATE SET
                           field_id = excluded.field_id,
                           label = excluded.label,
                           field_type = excluded.field_type,
                           reason = excluded.reason,
                           suggested_answer = CASE
                               WHEN excluded.suggested_answer <> '' THEN excluded.suggested_answer
                               ELSE blocked_questions.suggested_answer
                           END,
                           source = excluded.source,
                           updated_at = excluded.updated_at""",
                    (
                        job_id,
                        field_key,
                        field_id,
                        str(question.get("label", "")),
                        str(question.get("field_type", "")),
                        str(question.get("reason", "")),
                        str(question.get("suggested_answer", "")),
                        str(question.get("source", "")),
                        now,
                        now,
                    ),
                )
            rows = connection.execute(
                "SELECT * FROM blocked_questions WHERE job_id = ? ORDER BY id", (job_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def list_blocked_questions(self, job_id: int) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM blocked_questions WHERE job_id = ? ORDER BY id", (job_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def approve_blocked_question(self, question_id: int, answer: str, scope: str) -> dict:
        """Save a human-approved answer for this job, optionally for profile reuse."""
        if scope not in {"job_only", "profile_reusable"} or not answer.strip():
            raise ValueError("A non-empty answer and valid approval scope are required")
        now = self._timestamp()
        with self._connect() as connection:
            connection.execute(
                """UPDATE blocked_questions
                   SET approved_answer = ?, status = 'approved', scope = ?, updated_at = ?
                   WHERE id = ?""",
                (answer.strip(), scope, now, question_id),
            )
            row = connection.execute("SELECT * FROM blocked_questions WHERE id = ?", (question_id,)).fetchone()
        if row is None:
            raise ValueError(f"Blocked question {question_id} does not exist")
        return dict(row)

    def reject_blocked_question(self, question_id: int) -> dict:
        """Mark a portal question as intentionally skipped by the reviewer."""
        now = self._timestamp()
        with self._connect() as connection:
            connection.execute(
                "UPDATE blocked_questions SET status = 'rejected', updated_at = ? WHERE id = ?",
                (now, question_id),
            )
            row = connection.execute("SELECT * FROM blocked_questions WHERE id = ?", (question_id,)).fetchone()
        if row is None:
            raise ValueError(f"Blocked question {question_id} does not exist")
        return dict(row)

    def approved_answers_for_job(self, job_id: int) -> dict[str, str]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT field_key, approved_answer FROM blocked_questions
                   WHERE job_id = ? AND status = 'approved' AND approved_answer <> ''""",
                (job_id,),
            ).fetchall()
        return {row["field_key"]: row["approved_answer"] for row in rows}

    def mark_blocked_questions_filled(self, job_id: int, field_keys: Iterable[str]) -> None:
        """Close pending questions whose values were applied successfully to the portal."""
        keys = [key for key in field_keys if key]
        if not keys:
            return
        now = self._timestamp()
        placeholders = ", ".join("?" for _ in keys)
        with self._connect() as connection:
            connection.execute(
                f"""UPDATE blocked_questions SET status = 'filled', updated_at = ?
                    WHERE job_id = ? AND status = 'pending' AND field_key IN ({placeholders})""",
                (now, job_id, *keys),
            )

    def record_llm_call(
        self,
        *,
        job_id: int,
        purpose: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        payload_bytes: int,
        outcome: str,
        request_payload: str = "",
        response_content: str = "",
        duration_ms: int = 0,
    ) -> dict:
        """Persist one bounded LLM operation for cost and accuracy auditing."""
        if min(input_tokens, output_tokens, payload_bytes, duration_ms) < 0 or estimated_cost_usd < 0:
            raise ValueError("LLM usage values cannot be negative")
        now = self._timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO llm_calls
                                     (job_id, purpose, provider, model, input_tokens, output_tokens,
                                        estimated_cost_usd, payload_bytes, outcome, request_payload,
                                        response_content, duration_ms, created_at)
                                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, purpose, provider, model, input_tokens, output_tokens,
                                 estimated_cost_usd, payload_bytes, outcome, request_payload[:12_000],
                                 response_content[:12_000], duration_ms, now),
            )
            row = connection.execute("SELECT * FROM llm_calls WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)

    def list_llm_calls(self, job_id: int) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM llm_calls WHERE job_id = ? ORDER BY id", (job_id,)).fetchall()
        return [dict(row) for row in rows]

    def daily_llm_cost(self) -> float:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM llm_calls WHERE date(created_at) = date('now')"
            ).fetchone()
        return float(row[0])

    def can_spend_daily_llm_budget(self, daily_budget_usd: float, planned_cost_usd: float) -> bool:
        """Return whether an additional estimated operation fits the UTC daily budget."""
        if daily_budget_usd < 0 or planned_cost_usd < 0:
            raise ValueError("LLM budget values cannot be negative")
        return self.daily_llm_cost() + planned_cost_usd <= daily_budget_usd + 1e-9

    def register_resume_variant(self, variant_key: str, path: str) -> dict:
        """Register an approved resume file without copying its contents into storage."""
        if not Path(path).is_file():
            raise ValueError(f"Resume file does not exist: {path}")
        now = self._timestamp()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO resume_variants (variant_key, path, created_at, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(variant_key) DO UPDATE SET path = excluded.path, updated_at = excluded.updated_at""",
                (variant_key, path, now, now),
            )
        return self.get_resume_variant(variant_key)

    def get_resume_variant(self, variant_key: str) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM resume_variants WHERE variant_key = ?", (variant_key,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Resume variant {variant_key} does not exist")
        return dict(row)

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS application_runs (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS application_jobs (
                    id INTEGER PRIMARY KEY,
                    run_id INTEGER NOT NULL REFERENCES application_runs(id),
                    url TEXT NOT NULL,
                    current_url TEXT,
                    domain TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    checkpoint TEXT NOT NULL,
                    selected_resume TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(run_id, url)
                );
                CREATE TABLE IF NOT EXISTS application_events (
                    id INTEGER PRIMARY KEY,
                    job_id INTEGER NOT NULL REFERENCES application_jobs(id),
                    from_state TEXT,
                    to_state TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS resume_variants (
                    variant_key TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS llm_calls (
                    id INTEGER PRIMARY KEY,
                    job_id INTEGER NOT NULL REFERENCES application_jobs(id),
                    purpose TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    payload_bytes INTEGER NOT NULL,
                    outcome TEXT NOT NULL,
                    request_payload TEXT NOT NULL DEFAULT '',
                    response_content TEXT NOT NULL DEFAULT '',
                    duration_ms INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS blocked_questions (
                    id INTEGER PRIMARY KEY,
                    job_id INTEGER NOT NULL REFERENCES application_jobs(id),
                    field_key TEXT NOT NULL,
                    field_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    field_type TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    suggested_answer TEXT NOT NULL DEFAULT '',
                    approved_answer TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    scope TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(job_id, field_key)
                );
                """
            )
            existing_columns = {row[1] for row in connection.execute("PRAGMA table_info(llm_calls)")}
            for name, definition in (
                ("request_payload", "TEXT NOT NULL DEFAULT ''"),
                ("response_content", "TEXT NOT NULL DEFAULT ''"),
                ("duration_ms", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if name not in existing_columns:
                    connection.execute(f"ALTER TABLE llm_calls ADD COLUMN {name} {definition}")

    def _get_job(self, job_id: int) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM application_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise ValueError(f"Job {job_id} does not exist")
        return dict(row)

    @staticmethod
    def _allowed_next_states(current_state: JobState) -> set[JobState]:
        if current_state == JobState.QUEUED:
            return {JobState.OPENING_PORTAL, JobState.SKIPPED, JobState.FAILED_RETRYABLE, *BLOCKED_STATES}
        if current_state in ACTIVE_STATES:
            return ACTIVE_STATES | BLOCKED_STATES | {JobState.READY_FOR_REVIEW, JobState.SKIPPED}
        return set()

    @staticmethod
    def _unique_urls(application_urls: Iterable[str]) -> list[str]:
        unique_urls: list[str] = []
        for raw_url in application_urls:
            url = raw_url.strip()
            parsed_url = urlparse(url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                continue
            if url not in unique_urls:
                unique_urls.append(url)
        return unique_urls

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _refresh_run_status(connection: sqlite3.Connection, run_id: int, now: str) -> None:
        states = [row[0] for row in connection.execute("SELECT state FROM application_jobs WHERE run_id = ?", (run_id,))]
        status = "completed" if states and all(state in {item.value for item in TERMINAL_STATES} for state in states) else "active"
        connection.execute("UPDATE application_runs SET status = ?, updated_at = ? WHERE id = ?", (status, now, run_id))