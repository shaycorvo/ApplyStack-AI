"""Visible, review-first portal processing for one queued application job."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Awaitable, Callable

from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from browser_use.controller.service import Controller

from src.agent.application_policy import build_navigation_only_task, build_review_first_application_task
from src.agent.application_queue import ApplicationQueue, JobState, select_resume_variant
from src.agent.application_worker import ProcessingResult
from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.agent.form_resolution import (
    BudgetExceededError,
    CallLimitExceededError,
    FieldResolutionStatus,
    resolve_form_fields,
    resolve_unresolved_fields_with_llm,
)
from src.agent.portal_forms import (
    extract_form_fields,
    open_application_entry,
    portal_field_key,
    prepare_form_with_approved_answers,
    upload_resume_fields,
    wait_for_application_controls,
)
from src.profile.application_answers import LEGAL_TERMS, SENSITIVE_TERMS, save_reusable_narrative_answer
from src.browser.custom_browser import CustomBrowser
from src.controller.custom_controller import CustomController
from src.utils.llm_provider import get_llm_model
from src.utils.runtime_paths import data_path


PortalRunner = Callable[[dict, str], Awaitable[tuple[str, str]]]


def final_action_guard_script() -> str:
    """Return the page-init script that blocks finalization at the browser boundary."""
    return """
const finalAction = /\\b(submit|send application|confirm application|finish application)\\b/i;
document.addEventListener('click', (event) => {
    const element = event.target?.closest?.('button, input[type="submit"], a, [role="button"]');
    const label = `${element?.innerText || ''} ${element?.value || ''} ${element?.getAttribute('aria-label') || ''}`;
    if (finalAction.test(label)) {
        event.preventDefault();
        event.stopImmediatePropagation();
    }
}, true);
document.addEventListener('submit', (event) => {
    const submitter = event.submitter;
    const label = `${submitter?.innerText || ''} ${submitter?.value || ''} ${submitter?.getAttribute('aria-label') || ''}`;
    if (!submitter || finalAction.test(label)) {
        event.preventDefault();
        event.stopImmediatePropagation();
    }
}, true);
"""


async def install_final_action_guard(context) -> None:
    """Install the submission guard before navigating to a portal."""
    session = await context.get_session()
    await session.context.add_init_script(final_action_guard_script())


def extract_portal_outcome(report: str) -> str:
    """Select the agent's final observed outcome, excluding prior tool context."""
    normalized_report = report.lower()
    marker = "stopped because"
    marker_index = normalized_report.rfind(marker)
    if marker_index >= 0:
        return report[marker_index:]
    return report


def classify_portal_outcome(report: str, current_url: str) -> ProcessingResult:
    """Map an agent report to a conservative, persisted queue state."""
    outcome = extract_portal_outcome(report)
    normalized_report = outcome.lower()
    classifications = (
        (("captcha", "bot check", "human verification"), JobState.WAITING_CAPTCHA, "CAPTCHA or bot verification requires human action."),
        (("one-time password", "otp", "mfa", "verification code", "email verification", "sms verification"), JobState.WAITING_VERIFICATION, "Account verification requires human action."),
        (("log in", "login", "sign in", "sign-in", "create an account", "sign up"), JobState.WAITING_LOGIN, "An active portal session is required."),
        (("legal", "privacy policy", "terms", "consent", "attestation", "background check", "electronic signature"), JobState.WAITING_REVIEW, "A legal or consent decision requires human review."),
        (("submit application", "final review", "send application", "confirm application", "finish application"), JobState.WAITING_REVIEW, "The application reached a final action and is waiting for human review."),
        (("missing answer", "unknown answer", "required answer", "personal judgment"), JobState.WAITING_ANSWER, "A required answer needs an evidence-grounded human review."),
    )
    for signals, state, detail in classifications:
        if any(signal in normalized_report for signal in signals):
            return ProcessingResult(state=state, detail=detail, current_url=current_url, metadata={"report": outcome})
    return ProcessingResult(
        state=JobState.WAITING_REVIEW,
        detail="Portal preparation stopped for human review; no final action was taken.",
        current_url=current_url,
        metadata={"report": outcome},
    )


class ReviewFirstPortalProcessor:
    """Prepare one portal application in a visible browser and retain manual control."""

    def __init__(
        self,
        profile_path: str | Path | None = None,
        runner: PortalRunner | None = None,
        queue: ApplicationQueue | None = None,
    ):
        self.profile_path = Path(profile_path) if profile_path else data_path("profile", "profile.json")
        self.runner = runner or self._run_browser_agent
        self.queue = queue
        self._active_agents: list[BrowserUseAgent] = []

    def __call__(self, job: dict) -> ProcessingResult:
        resume_path = self._select_resume_path(job)
        if os.getenv("APPLYAGENT_TEST_MODE", "false").lower() in {"1", "true", "yes"}:
            return ProcessingResult(
                state=JobState.WAITING_REVIEW,
                detail="Test mode prepared the application locally; no portal, LLM, or submission action was used.",
                current_url=job["url"],
                selected_resume=resume_path,
                metadata={"test_mode": True, "submitted": False, "llm_calls": 0},
            )
        task = build_review_first_application_task(job["url"])
        task += f"\n\nUse this approved resume when a resume upload is requested: {resume_path}"
        report, current_url = asyncio.run(self.runner(job, task))
        result = classify_portal_outcome(report, current_url or job["url"])
        return ProcessingResult(
            state=result.state,
            detail=result.detail,
            current_url=result.current_url,
            selected_resume=resume_path,
            metadata=result.metadata,
        )

    def _select_resume_path(self, job: dict) -> str:
        if not self.profile_path.is_file():
            raise ValueError("Approved profile is required before portal preparation")
        profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        decision = select_resume_variant(job.get("title", ""), job.get("job_description", ""))
        documents = profile.get("documents", {})
        resume_paths = documents.get("resume_routing", {})
        candidates = (
            resume_paths.get(decision.variant_key),
            resume_paths.get("default"),
            documents.get("resume_path"),
        )
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return candidate
        raise ValueError(f"Approved {decision.variant_key} resume is not available")

    async def _run_browser_agent(self, job: dict, task: str) -> tuple[str, str]:
        browser = CustomBrowser(config=BrowserConfig(headless=False, keep_alive=True))
        await browser.async_start()
        context = await browser.create_context(
            BrowserContextConfig(window_width=1280, window_height=1024, keep_alive=True)
        )
        await install_final_action_guard(context)
        await context.navigate_to(job["url"])
        page = await context.get_current_page()
        await wait_for_application_controls(page)
        profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        fields = await extract_form_fields(page)
        if not fields and await open_application_entry(page):
            await wait_for_application_controls(page)
            fields = await extract_form_fields(page)

        if fields:
            uploaded_field_ids = await upload_resume_fields(page, fields, self._select_resume_path(job))
            approved_answers = dict(profile.get("application_answers", {}).get("reusable_narrative_answers", {}))
            if self.queue:
                approved_answers.update(self.queue.approved_answers_for_job(job["id"]))
            preparation = await prepare_form_with_approved_answers(page, fields, profile, approved_answers)
            local_resolutions = resolve_form_fields(fields, profile)
            unresolved_fields = [
                field for field in fields if field["field_id"] in preparation.unresolved_field_ids and field["field_id"] not in uploaded_field_ids
            ]
            llm_resolutions = self._resolve_eligible_questions(job, profile, unresolved_fields)
            llm_answers = {
                portal_field_key(field): resolution.value
                for field in unresolved_fields
                if (resolution := next((item for item in llm_resolutions if item.field_id == field["field_id"]), None))
                and resolution.status == FieldResolutionStatus.RESOLVED
                and resolution.value
            }
            if llm_answers:
                llm_preparation = await prepare_form_with_approved_answers(page, unresolved_fields, profile, llm_answers)
                unresolved_field_by_id = {field["field_id"]: field for field in unresolved_fields}
                filled_answer_keys = {
                    portal_field_key(field)
                    for field in unresolved_fields
                    if field["field_id"] in llm_preparation.filled_field_ids
                }
                preparation = type(preparation)(
                    filled_field_ids=[*preparation.filled_field_ids, *llm_preparation.filled_field_ids],
                    unresolved_field_ids=[
                        field_id
                        for field_id in preparation.unresolved_field_ids
                        if field_id not in unresolved_field_by_id
                        or portal_field_key(unresolved_field_by_id[field_id]) not in filled_answer_keys
                    ],
                )
                for field in unresolved_fields:
                    if (
                        field["field_id"] in llm_preparation.filled_field_ids
                        and str(field.get("type", "")).lower() == "radio"
                        and llm_answers.get(portal_field_key(field)) == "Yes"
                    ):
                        save_reusable_narrative_answer(self.profile_path, portal_field_key(field), "Yes")
            resolution_by_id = {resolution.field_id: resolution for resolution in [*local_resolutions, *llm_resolutions]}
            if self.queue:
                self.queue.mark_blocked_questions_filled(
                    job["id"],
                    [
                        portal_field_key(field)
                        for field in fields
                        if field["field_id"] in preparation.filled_field_ids
                    ],
                )
            unresolved_fields = [
                field for field in unresolved_fields if field["field_id"] in preparation.unresolved_field_ids
            ]
            self._record_blocked_questions(job, unresolved_fields, resolution_by_id)
            unresolved_labels = [
                field["label"] or field["field_id"]
                for field in unresolved_fields
            ]
            if unresolved_labels:
                return (
                    "Stopped because required answer needs human review: " + ", ".join(unresolved_labels[:8]),
                    page.url,
                )
            return "Stopped because standard approved fields were filled and the form is ready for human review.", page.url

        if os.getenv("ENABLE_NAVIGATION_LLM_FALLBACK", "false").lower() not in {"1", "true", "yes"}:
            return "Stopped because no application form or known safe application entry control was found.", page.url

        provider = os.getenv("LLM_PROVIDER", "google")
        model_name = os.getenv("LLM_MODEL", "gemini-3.6-flash")
        llm = get_llm_model(provider=provider, model_name=model_name, temperature=0.0)
        agent = BrowserUseAgent(
            task=build_navigation_only_task(job["url"]),
            llm=llm,
            browser_context=context,
            controller=Controller(),
            enable_memory=False,
            max_actions_per_step=3,
            max_input_tokens=8000,
        )
        agent.keep_browser_open = True
        self._active_agents.append(agent)
        history = await agent.run(max_steps=4)
        current_url = job["url"]
        report_parts: list[str] = []
        for entry in getattr(history, "history", []):
            state = getattr(entry, "state", None)
            current_url = getattr(state, "url", current_url) or current_url
            for action_result in getattr(entry, "result", []) or []:
                report_parts.append(str(getattr(action_result, "extracted_content", "") or getattr(action_result, "error", "")))
        return "\n".join(part for part in report_parts if part)[-12000:], current_url

    def _resolve_eligible_questions(self, job: dict, profile: dict, fields: list[dict]) -> list:
        if not self.queue:
            return []
        eligible_fields = [field for field in fields if _is_llm_eligible_field(field)]
        if not eligible_fields:
            return []
        provider = os.getenv("LLM_PROVIDER", "google")
        model_name = os.getenv("LLM_MODEL", "gemini-3.6-flash")
        try:
            return resolve_unresolved_fields_with_llm(
                llm=get_llm_model(provider=provider, model_name=model_name, temperature=0.0),
                queue=self.queue,
                job=job,
                profile=profile,
                fields=eligible_fields,
                provider=provider,
                model=model_name,
                daily_budget_usd=float(os.getenv("LLM_DAILY_BUDGET_USD", "1.00")),
                input_cost_per_million=float(os.getenv("LLM_INPUT_COST_PER_MILLION", "0")),
                output_cost_per_million=float(os.getenv("LLM_OUTPUT_COST_PER_MILLION", "0")),
            )
        except (BudgetExceededError, CallLimitExceededError, ValueError):
            return []

    def _record_blocked_questions(self, job: dict, fields: list[dict], resolutions: dict) -> None:
        if not self.queue or not fields:
            return
        self.queue.record_blocked_questions(
            job["id"],
            [
                {
                    "field_key": portal_field_key(field),
                    "field_id": field["field_id"],
                    "label": field["label"] or field["field_id"],
                    "field_type": field.get("type", ""),
                    "reason": getattr(resolutions.get(field["field_id"]), "reason", "This portal field needs human review."),
                    "suggested_answer": getattr(resolutions.get(field["field_id"]), "value", None) or "",
                    "source": getattr(resolutions.get(field["field_id"]), "source", None) or "",
                }
                for field in fields
            ],
        )


def _is_llm_eligible_field(field: dict) -> bool:
    """Allow factual drafts and user-approved routine Yes/No eligibility answers."""
    field_type = str(field.get("type", "")).lower()
    if field_type not in {"text", "textarea", "radio"}:
        return False
    label = str(field.get("label", "")).lower()
    excluded_terms = (*SENSITIVE_TERMS, *LEGAL_TERMS, "salary", "compensation", "notice period", "availability", "relocate")
    if any(term in label for term in excluded_terms):
        return False
    if field_type == "radio":
        return str(field.get("option_label", "")).strip().casefold() == "yes"
    return len(label) >= 30 and ("?" in label or label.startswith(("why ", "what ", "how ")))