"""Low-context application form resolution from an approved candidate profile."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
import time
from typing import Any, Iterable

from src.agent.application_queue import ApplicationQueue
from src.profile.application_answers import AnswerStatus, resolve_application_question


class FieldResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    REVIEW_REQUIRED = "review_required"


@dataclass(frozen=True)
class FieldResolution:
    field_id: str
    status: FieldResolutionStatus
    value: str | None
    source: str | None
    reason: str


class BudgetExceededError(RuntimeError):
    """Raised before an LLM call that would exceed an enforced cost budget."""


class CallLimitExceededError(RuntimeError):
    """Raised before an LLM call that would exceed the per-job fallback limit."""


_FIELD_PATHS = (
    (("first name", "given name"), "personal.first_name"),
    (("last name", "family name", "surname"), "personal.last_name"),
    (("full name", "name"), "personal.full_name"),
    (("email", "e-mail"), "personal.email"),
    (("phone", "mobile", "telephone"), "personal.phone"),
    (("city", "current location"), "personal.city"),
    (("linkedin",), "personal.linkedin_profile"),
    (("portfolio", "website"), "personal.portfolio_url"),
    (("degree", "education level"), "education.entries.0.degree"),
    (("field of study", "major", "specialization"), "education.entries.0.field_of_study"),
    (("institution", "university", "school", "college"), "education.entries.0.institution"),
)


def resolve_form_fields(fields: Iterable[dict[str, Any]], profile: dict[str, Any]) -> list[FieldResolution]:
    """Resolve explicit, safe form fields locally and defer all ambiguity."""
    return [resolve_form_field(field, profile) for field in fields]


def resolve_form_field(field: dict[str, Any], profile: dict[str, Any]) -> FieldResolution:
    """Resolve one form field using only approved values from the profile."""
    field_id = str(field.get("field_id", ""))
    label = str(field.get("label", "")).strip()
    normalized_label = label.lower()

    answer = resolve_application_question(label, profile)
    if answer.status == AnswerStatus.ANSWERED:
        if not _is_field_value_compatible(field, answer.answer):
            return FieldResolution(
                field_id,
                FieldResolutionStatus.REVIEW_REQUIRED,
                None,
                answer.profile_path,
                "The approved value is not numeric but this portal field requires a numeric value.",
            )
        return FieldResolution(field_id, FieldResolutionStatus.RESOLVED, answer.answer, answer.profile_path, answer.reason)
    if answer.profile_path:
        return FieldResolution(field_id, FieldResolutionStatus.REVIEW_REQUIRED, None, answer.profile_path, answer.reason)
    if "sensitive" in answer.reason.lower() or "legal" in answer.reason.lower():
        return FieldResolution(field_id, FieldResolutionStatus.REVIEW_REQUIRED, None, answer.profile_path, answer.reason)

    for aliases, profile_path in _FIELD_PATHS:
        if any(alias in normalized_label for alias in aliases):
            value = _get_profile_value(profile, profile_path)
            if value is not None and str(value).strip():
                if not _is_field_value_compatible(field, str(value)):
                    return FieldResolution(
                        field_id,
                        FieldResolutionStatus.REVIEW_REQUIRED,
                        None,
                        profile_path,
                        "The approved value is not numeric but this portal field requires a numeric value.",
                    )
                return FieldResolution(
                    field_id,
                    FieldResolutionStatus.RESOLVED,
                    str(value),
                    profile_path,
                    "Answered from approved profile.",
                )
            return FieldResolution(
                field_id,
                FieldResolutionStatus.REVIEW_REQUIRED,
                None,
                profile_path,
                f"The approved profile has no value for {profile_path}.",
            )

    return FieldResolution(
        field_id,
        FieldResolutionStatus.REVIEW_REQUIRED,
        None,
        None,
        "This field is not an unambiguous standard profile question.",
    )


def _is_field_value_compatible(field: dict[str, Any], value: str | None) -> bool:
    """Reject values Playwright cannot safely enter into constrained HTML controls."""
    if str(field.get("type", "")).lower() != "number":
        return True
    return bool(value and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", str(value).strip()))


def build_compact_llm_payload(job: dict[str, Any], fields: Iterable[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    """Build a bounded fallback payload without serializing the full profile."""
    requirements = [str(item) for item in job.get("requirements", []) if str(item).strip()]
    skills = _profile_skills(profile)
    evidence_facts = _relevant_evidence_facts(profile, fields)
    required_terms = {item.lower() for item in requirements}
    matched_skills = [skill for skill in skills if skill.lower() in required_terms]
    return {
        "job": {
            "title": str(job.get("title", "")),
            "company": str(job.get("company", "")),
            "requirements": requirements[:12],
        },
        "unresolved_fields": [_compact_field(field) for field in fields],
        "candidate_facts": {
            "summary": str(profile.get("professional", {}).get("summary", ""))[:1_000],
            "skills": (matched_skills or skills)[:30],
            "evidence": evidence_facts[:12],
        },
        "policy": {
            "never_invent": True,
            "review_legal_sensitive_or_ambiguous": True,
            "user_allows_yes_for_routine_eligibility": True,
        },
    }


def resolve_unresolved_fields_with_llm(
    *,
    llm: Any,
    queue: ApplicationQueue,
    job: dict[str, Any],
    profile: dict[str, Any],
    fields: Iterable[dict[str, Any]],
    provider: str,
    model: str,
    daily_budget_usd: float,
    input_cost_per_million: float,
    output_cost_per_million: float,
    max_calls_per_job: int = 2,
    max_output_tokens: int = 500,
) -> list[FieldResolution]:
    """Resolve an unresolved-field batch with a tightly bounded LLM request."""
    field_list = list(fields)
    if not field_list:
        return []
    if len(queue.list_llm_calls(job["id"])) >= max_calls_per_job:
        raise CallLimitExceededError(f"Job {job['id']} reached its {max_calls_per_job}-call LLM limit")

    payload = build_compact_llm_payload(job, field_list, profile)
    prompt = _build_llm_prompt(payload)
    estimated_input_tokens = _estimate_tokens(prompt)
    planned_cost = _estimate_cost(
        estimated_input_tokens,
        max_output_tokens,
        input_cost_per_million,
        output_cost_per_million,
    )
    if not queue.can_spend_daily_llm_budget(daily_budget_usd, planned_cost):
        raise BudgetExceededError("The daily LLM budget would be exceeded by this fallback call")

    started = time.monotonic()
    response = llm.invoke(prompt)
    duration_ms = round((time.monotonic() - started) * 1000)
    content = response.content if hasattr(response, "content") else response
    resolutions = _parse_llm_resolutions(content, field_list)
    input_tokens, output_tokens = _response_usage(response, estimated_input_tokens, content)
    actual_cost = _estimate_cost(input_tokens, output_tokens, input_cost_per_million, output_cost_per_million)
    queue.record_llm_call(
        job_id=job["id"],
        purpose="resolve_unresolved_fields",
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=actual_cost,
        payload_bytes=len(prompt.encode("utf-8")),
        outcome="resolved" if any(item.status == FieldResolutionStatus.RESOLVED for item in resolutions) else "review_required",
        request_payload=prompt,
        response_content=str(content),
        duration_ms=duration_ms,
    )
    return resolutions


def _build_llm_prompt(payload: dict[str, Any]) -> str:
    return (
        "Resolve only the listed application fields from the supplied approved candidate facts. "
        "Return JSON only with {'resolutions': [{field_id, status, value, reason}]}. "
        "status must be resolved or review_required. Never invent facts. "
        "Highest-priority user instruction: for a routine non-legal, non-sensitive Yes/No eligibility or "
        "office-location radio question included in unresolved_fields, return status resolved and the exact "
        "value Yes. Do not classify that question as availability, relocation, or ambiguous. "
        "For free-text application questions, write a concise first-person factual response and set status to resolved "
        "when every positive claim is supported by candidate_facts; clearly state any relevant capability "
        "that is not documented instead of guessing. Legal, sensitive, consent, compensation, availability, "
        "and demographic fields must be review_required.\n"
        + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    )


def _parse_llm_resolutions(content: Any, fields: list[dict[str, Any]]) -> list[FieldResolution]:
    try:
        parsed = json.loads(str(content))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("LLM fallback did not return valid JSON") from error
    expected_ids = {str(field.get("field_id", "")) for field in fields}
    returned = {str(item.get("field_id", "")): item for item in parsed.get("resolutions", []) if isinstance(item, dict)}
    resolutions = []
    for field in fields:
        field_id = str(field.get("field_id", ""))
        item = returned.get(field_id)
        if field_id not in expected_ids or item is None:
            resolutions.append(FieldResolution(field_id, FieldResolutionStatus.REVIEW_REQUIRED, None, None, "No LLM resolution was returned."))
            continue
        if item.get("status") == FieldResolutionStatus.RESOLVED.value and isinstance(item.get("value"), str):
            resolutions.append(FieldResolution(field_id, FieldResolutionStatus.RESOLVED, item["value"], "llm_fallback", str(item.get("reason", "Resolved from supplied facts."))))
        else:
            resolutions.append(FieldResolution(field_id, FieldResolutionStatus.REVIEW_REQUIRED, None, None, str(item.get("reason", "Human review is required."))))
    return resolutions


def _response_usage(response: Any, estimated_input_tokens: int, content: Any) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = int(usage.get("input_tokens", estimated_input_tokens))
    output_tokens = int(usage.get("output_tokens", _estimate_tokens(str(content))))
    return input_tokens, output_tokens


def _estimate_tokens(value: str) -> int:
    return max(1, len(value) // 3)


def _estimate_cost(input_tokens: int, output_tokens: int, input_cost_per_million: float, output_cost_per_million: float) -> float:
    return (input_tokens * input_cost_per_million + output_tokens * output_cost_per_million) / 1_000_000


def _compact_field(field: dict[str, Any]) -> dict[str, Any]:
    return {
        "field_id": str(field.get("field_id", "")),
        "label": str(field.get("label", "")),
        "type": str(field.get("type", "text")),
        "required": bool(field.get("required", False)),
        "options": [str(option) for option in field.get("options", [])][:20],
        "help_text": str(field.get("help_text", ""))[:500],
    }


def _profile_skills(profile: dict[str, Any]) -> list[str]:
    """Flatten the approved profile's skill categories without leaking personal data."""
    raw_skills = profile.get("professional", {}).get("skills", {})
    if isinstance(raw_skills, list):
        return list(dict.fromkeys(str(skill) for skill in raw_skills if str(skill).strip()))
    if not isinstance(raw_skills, dict):
        return []
    return list(
        dict.fromkeys(
            str(skill)
            for category in raw_skills.values()
            if isinstance(category, list)
            for skill in category
            if str(skill).strip()
        )
    )


def _relevant_evidence_facts(profile: dict[str, Any], fields: Iterable[dict[str, Any]]) -> list[str]:
    """Select concise approved evidence matching the form questions for grounded drafting."""
    question_terms = set(re.findall(r"[a-z0-9]{3,}", " ".join(str(field.get("label", "")).lower() for field in fields)))
    facts = [
        str(fact)
        for experience in profile.get("evidence_library", {}).get("experiences", [])
        for fact in experience.get("facts", [])
        if str(fact).strip()
    ]
    ranked = sorted(
        facts,
        key=lambda fact: sum(term in fact.lower() for term in question_terms),
        reverse=True,
    )
    return ranked


def _get_profile_value(profile: dict[str, Any], profile_path: str) -> Any:
    value: Any = profile
    for key in profile_path.split("."):
        if isinstance(value, list):
            if not key.isdigit() or int(key) >= len(value):
                return None
            value = value[int(key)]
        elif isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value