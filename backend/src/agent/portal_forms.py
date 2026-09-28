"""Direct Playwright form inspection and safe deterministic field filling."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from src.agent.form_resolution import FieldResolutionStatus, resolve_form_fields


@dataclass(frozen=True)
class FormPreparation:
    filled_field_ids: list[str]
    unresolved_field_ids: list[str]


async def wait_for_application_controls(page: Any, timeout_ms: int = 10_000) -> None:
    """Wait briefly for client-rendered application controls without clicking anything."""
    try:
        await page.wait_for_function(
            """() => document.querySelector(
                'input:not([type="hidden"]):not([type="submit"]), textarea, select, button, a'
            )""",
            timeout=timeout_ms,
        )
    except Exception:
        # The caller still classifies an empty portal conservatively for review.
        return


async def extract_form_fields(page: Any) -> list[dict[str, Any]]:
    """Read actionable controls only, excluding navigation and final-submit elements."""
    return await page.evaluate(
        """() => {
            const elements = [...document.querySelectorAll('input, textarea, select')]
                .filter((element) => !['hidden', 'submit', 'button', 'reset'].includes(element.type));
            return elements.map((element, index) => {
                const fieldId = `applyagent-${index}`;
                element.dataset.applyagentFieldId = fieldId;
                const labels = element.labels ? [...element.labels].map((label) => label.innerText.trim()).filter(Boolean) : [];
                let containerText = '';
                const choiceLabel = labels[0] || element.getAttribute('aria-label') || element.placeholder || element.name || '';
                let ancestor = element.parentElement;
                for (let depth = 0; ancestor && depth < 5; depth += 1, ancestor = ancestor.parentElement) {
                    const text = (ancestor.innerText || '').trim().replace(/\\s+/g, ' ');
                    const radioQuestion = element.type === 'radio' && text !== choiceLabel && text.length > choiceLabel.length + 8;
                    if (text && text.length <= 500 && (element.type !== 'radio' || radioQuestion)) {
                        containerText = text;
                        break;
                    }
                }
                const label = element.type === 'radio' && containerText ? containerText : choiceLabel || containerText;
                return {
                    field_id: fieldId,
                    label,
                    type: element.tagName.toLowerCase() === 'select' ? 'select' : (element.type || element.tagName.toLowerCase()),
                    option_label: element.type === 'radio' ? choiceLabel : '',
                    required: element.required,
                    options: element.tagName.toLowerCase() === 'select' ? [...element.options].map((option) => option.text.trim()).filter(Boolean) : [],
                    help_text: element.getAttribute('aria-describedby') || '',
                    selector: `[data-applyagent-field-id="${fieldId}"]`,
                };
            });
        }"""
    )


async def open_application_entry(page: Any) -> bool:
    """Open a non-final application entry page when a known safe control is visible."""
    entry_pattern = re.compile(r"^(apply now|start application|continue)$", re.IGNORECASE)
    for role in ("button", "link"):
        locator = page.get_by_role(role, name=entry_pattern)
        if await locator.count():
            await locator.first.click()
            await page.wait_for_load_state("domcontentloaded")
            return True
    return False


async def prepare_standard_form(page: Any, fields: list[dict[str, Any]], profile: dict[str, Any]) -> FormPreparation:
    """Fill only deterministic, approved values and leave all other controls unchanged."""
    return await prepare_form_with_approved_answers(page, fields, profile, {})


async def prepare_form_with_approved_answers(
    page: Any,
    fields: list[dict[str, Any]],
    profile: dict[str, Any],
    approved_answers: dict[str, str],
) -> FormPreparation:
    """Fill deterministic values plus answers explicitly approved for this retry."""
    filled_field_ids: list[str] = []
    unresolved_field_ids: list[str] = []
    for field, resolution in zip(fields, resolve_form_fields(fields, profile)):
        approved_value = approved_answers.get(portal_field_key(field))
        value = approved_value or (resolution.value if resolution.status == FieldResolutionStatus.RESOLVED else None)
        if not value:
            unresolved_field_ids.append(resolution.field_id)
            continue
        if str(field.get("type", "")).lower() == "radio" and approved_value and not _matches_radio_option(field, approved_value):
            continue
        if await _fill_field(page, field, value):
            filled_field_ids.append(resolution.field_id)
        else:
            unresolved_field_ids.append(resolution.field_id)
    return FormPreparation(filled_field_ids, unresolved_field_ids)


def portal_field_key(field: dict[str, Any]) -> str:
    """Create a stable question key because portal element ids change between visits."""
    label = str(field.get("label", "")).lower().strip()
    return re.sub(r"\s+", " ", label)[:500]


async def upload_resume_fields(page: Any, fields: list[dict[str, Any]], resume_path: str) -> list[str]:
    """Upload an approved resume only to a field explicitly identified as a resume or CV."""
    uploaded_field_ids: list[str] = []
    for field in fields:
        if not _is_resume_file_field(field):
            continue
        selector = field.get("selector")
        if not selector:
            continue
        try:
            await page.locator(selector).set_input_files(resume_path)
        except Exception:
            continue
        uploaded_field_ids.append(field["field_id"])
    return uploaded_field_ids


def _is_resume_file_field(field: dict[str, Any]) -> bool:
    """Reject portfolios and arbitrary attachments; only resume/CV uploads are automatic."""
    if str(field.get("type", "")).lower() != "file":
        return False
    label = str(field.get("label", "")).lower()
    return bool(re.search(r"\b(resume|curriculum vitae|cv)\b", label))


def _matches_radio_option(field: dict[str, Any], value: str) -> bool:
    return str(field.get("option_label", "")).strip().casefold() == value.strip().casefold()


async def _fill_field(page: Any, field: dict[str, Any], value: str) -> bool:
    """Use stable extracted selectors and never act on consent or submit controls."""
    field_type = str(field.get("type", "")).lower()
    if field_type in {"checkbox", "file", "hidden", "submit", "button", "reset"}:
        return False
    selector = field.get("selector")
    if not selector:
        return False
    locator = page.locator(selector)
    if field_type == "radio":
        if not _matches_radio_option(field, value):
            return False
        await locator.check()
        return True
    if field_type == "select":
        try:
            await locator.select_option(label=value)
        except Exception:
            return False
    else:
        await locator.fill(value)
    return True