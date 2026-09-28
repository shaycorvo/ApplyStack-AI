"""Deterministic, review-first answers for standard application questions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
import json
from pathlib import Path
from typing import Any


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    REQUIRES_REVIEW = "requires_review"


@dataclass(frozen=True)
class AnswerDecision:
    status: AnswerStatus
    answer: str | None
    reason: str
    profile_path: str | None = None


SENSITIVE_TERMS = ("gender", "race", "ethnicity", "veteran", "disability", "date of birth", "criminal")
LEGAL_TERMS = ("agree", "consent", "attest", "signature", "privacy policy", "terms", "background check")
STANDARD_QUESTIONS = (
    (("expected ctc", "expected salary", "expected compensation"), "application_answers.deterministic_fields.expected_ctc"),
    (("current ctc", "current salary", "current compensation"), "application_answers.deterministic_fields.current_ctc"),
    (("other compensation", "compensation benefits"), "application_answers.deterministic_fields.other_compensation_benefits"),
    (("education start date", "course start date", "start date"), "education.entries.0.start_date"),
    (("education end date", "course end date", "graduation date", "end date"), "education.entries.0.end_date"),
    (("visa sponsorship", "require sponsorship"), "preferences.visa_sponsorship"),
    (("authorized to work", "work authorization"), "preferences.work_authorization"),
    (("notice period", "when can you start", "availability"), "preferences.notice_period"),
    (("reason for change", "why are you leaving", "reason for leaving"), "application_answers.reason_for_change"),
    (("email",), "personal.email"),
    (("phone", "mobile number", "telephone"), "personal.phone"),
    (("city", "current location", "where are you located"), "personal.city"),
)


def resolve_application_question(question: str, profile: dict[str, Any]) -> AnswerDecision:
    """Return only unambiguous profile-backed answers; all other prompts need review."""
    normalized_question = question.lower()
    if any(term in normalized_question for term in SENSITIVE_TERMS):
        return AnswerDecision(AnswerStatus.REQUIRES_REVIEW, None, "This is a sensitive personal question.")
    if any(term in normalized_question for term in LEGAL_TERMS):
        return AnswerDecision(AnswerStatus.REQUIRES_REVIEW, None, "This is a legal or consent question.")

    for phrases, profile_path in STANDARD_QUESTIONS:
        if any(phrase in normalized_question for phrase in phrases):
            value = _get_profile_value(profile, profile_path)
            if profile_path == "application_answers.deterministic_fields.current_ctc" and not _is_positive_number(value):
                value = _get_profile_value(profile, "professional.current_salary")
                profile_path = "professional.current_salary"
            if value is not None:
                return AnswerDecision(AnswerStatus.ANSWERED, str(value), "Answered from approved profile.", profile_path)
            return AnswerDecision(
                AnswerStatus.REQUIRES_REVIEW,
                None,
                f"The approved profile has no value for {profile_path}.",
                profile_path,
            )
    return AnswerDecision(
        AnswerStatus.REQUIRES_REVIEW,
        None,
        "This requires an evidence-grounded draft and human review.",
    )


def save_deterministic_answers(profile_path: str | Path, answers: dict[str, str]) -> dict:
    """Persist human-approved values so equivalent future portal questions are automatic."""
    required_keys = {
        "education_start_date",
        "education_end_date",
        "current_ctc",
        "expected_ctc",
        "other_compensation_benefits",
    }
    if set(answers) != required_keys or any(not str(value).strip() for value in answers.values()):
        raise ValueError("All deterministic answer fields must be explicitly approved")
    for key in ("education_start_date", "education_end_date"):
        try:
            date.fromisoformat(answers[key])
        except ValueError as error:
            raise ValueError(f"{key} must use YYYY-MM-DD") from error

    path = Path(profile_path)
    profile = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    education = profile.setdefault("education", {})
    entries = education.setdefault("entries", [{}])
    if not entries:
        entries.append({})
    entries[0]["start_date"] = answers["education_start_date"]
    entries[0]["end_date"] = answers["education_end_date"]
    deterministic_fields = profile.setdefault("application_answers", {}).setdefault("deterministic_fields", {})
    deterministic_fields.update(
        {
            "current_ctc": answers["current_ctc"],
            "expected_ctc": answers["expected_ctc"],
            "other_compensation_benefits": answers["other_compensation_benefits"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def save_reusable_narrative_answer(profile_path: str | Path, question_key: str, answer: str) -> dict:
    """Persist a human-approved narrative answer under its normalized portal question."""
    if not question_key.strip() or not answer.strip():
        raise ValueError("A question and non-empty approved answer are required")
    path = Path(profile_path)
    profile = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    reusable_answers = profile.setdefault("application_answers", {}).setdefault("reusable_narrative_answers", {})
    reusable_answers[question_key] = answer.strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def _get_profile_value(profile: dict[str, Any], profile_path: str) -> Any:
    value: Any = profile
    for key in profile_path.split("."):
        if isinstance(value, list):
            if not key.isdigit() or int(key) >= len(value):
                return None
            value = value[int(key)]
        elif not isinstance(value, dict):
            return None
        else:
            value = value.get(key)
    return value


def _is_positive_number(value: Any) -> bool:
    """Treat empty and zero compensation overrides as unset."""
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False