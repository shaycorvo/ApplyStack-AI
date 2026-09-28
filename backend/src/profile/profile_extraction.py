"""Structured, evidence-grounded LLM profile extraction."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from json_repair import repair_json
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceItem(BaseModel):
    id: str
    topic: str
    facts: list[str]
    evidence_refs: list[str]

    @model_validator(mode="before")
    @classmethod
    def normalize_claim(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "claim" not in value:
            return value
        claim = value["claim"]
        if not isinstance(claim, str) or not claim.strip():
            return value
        return {
            "id": f"evidence-{sha256(claim.encode('utf-8')).hexdigest()[:12]}",
            "topic": value.get("topic", "experience"),
            "facts": value.get("facts", [claim]),
            "evidence_refs": value.get("evidence_refs", []),
        }


class ProfileDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    personal: dict[str, Any] = Field(default_factory=dict)
    professional: dict[str, Any] = Field(default_factory=dict)
    education: dict[str, Any] = Field(default_factory=dict)
    evidence_library: dict[str, list[EvidenceItem]] = Field(default_factory=lambda: {"experiences": []})
    clarification_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_education_entries(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not isinstance(value.get("education"), list):
            return value
        normalized_value = value.copy()
        normalized_value["education"] = {"entries": value["education"]}
        return normalized_value


def build_profile_extraction_prompt(sources: list[dict]) -> str:
    """Create a bounded extraction prompt from deterministic, persisted sources."""
    source_blocks = []
    for source in sources:
        if not source.get("extracted_text"):
            continue
        source_blocks.append(
            f"SOURCE {source['source_ref']} ({source['source_type']}: {source['file_name']})\n"
            f"{source['extracted_text']}"
        )
    if not source_blocks:
        raise ValueError("At least one locally extracted document is required")
    return """
Extract a candidate profile from the supplied sources. Return exactly one valid
JSON object with the keys personal, professional, education, evidence_library,
and clarification_questions. Do not wrap it in Markdown or return YAML. Do not
invent, infer, embellish, or combine facts; do not invent claims that are not
explicitly supported.
Do not produce credentials, legal answers,
EEO data, salary expectations, date of birth, or contact details unless the source
explicitly contains them. Every evidence_library item must include evidence_refs
using the exact supplied source reference. Put any missing, conflicting, unclear,
or sensitive fact into clarification_questions instead of guessing.

Organize the draft for job-application use: personal, professional, education,
and evidence_library.experiences. Every experience must contain id, topic, facts
(a list of short factual claims), and evidence_refs. Evidence facts must be short
factual claims
that can later support an application answer.
Education must be an object; put multiple credentials in education.entries.

SOURCES:
""".strip() + "\n\n" + "\n\n".join(source_blocks)


def extract_profile_draft(llm: Any, sources: list[dict]) -> dict:
    """Ask an LLM for a validated structured draft; the caller controls persistence."""
    result = llm.invoke(build_profile_extraction_prompt(sources))
    content = result.content if hasattr(result, "content") else result
    if not isinstance(content, str):
        raise ValueError("Profile extraction returned a non-text response")
    try:
        draft = json.loads(repair_json(content))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Profile extraction did not return a valid JSON object") from error
    return validate_reviewed_profile_draft(draft, {source["source_ref"] for source in sources})


def is_gemini_quota_error(error: Exception) -> bool:
    """Identify the provider's request-rate or daily quota response."""
    message = str(error).lower()
    return "429" in message and ("quota" in message or "rate limit" in message)


def is_openrouter_auth_error(error: Exception) -> bool:
    """Identify an invalid, revoked, or unavailable OpenRouter credential."""
    message = str(error).lower()
    return "401" in message and ("user not found" in message or "unauthorized" in message or "invalid api key" in message)


def validate_reviewed_profile_draft(draft: dict, allowed_evidence_refs: set[str]) -> dict:
    """Canonicalize a reviewed draft and ensure every claim remains traceable."""
    validated_draft = ProfileDraft.model_validate(draft)
    for category in validated_draft.evidence_library.values():
        for evidence in category:
            normalized_refs = []
            for evidence_ref in evidence.evidence_refs:
                normalized_ref = next(
                    (
                        allowed_ref
                        for allowed_ref in allowed_evidence_refs
                        if evidence_ref.startswith(f"{allowed_ref} (") and evidence_ref.endswith(")")
                    ),
                    evidence_ref,
                )
                normalized_refs.append(normalized_ref)
            evidence.evidence_refs = normalized_refs
            for evidence_ref in normalized_refs:
                if evidence_ref not in allowed_evidence_refs:
                    raise ValueError(f"Draft contains an unknown evidence reference: {evidence_ref}")
    return validated_draft.model_dump()