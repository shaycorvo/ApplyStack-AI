"""Policies shared by job-application automation entry points."""


def build_review_first_application_task(application_url: str) -> str:
    """Build a portal-agnostic task that prepares, but never submits, an application."""
    return f"""
You are preparing a job application in a visible browser for human review.

Open this application URL: {application_url}

First call get_profile() and use only verified information from that profile.
Fill standard application fields and upload the configured resume when requested.
Do not invent qualifications, employment history, answers, or personal information.

Inspect each page before acting. You may click any page-specific, non-final
navigation control needed to reach the application form, including controls such
as Apply now, Start application, Continue, Next, Create profile, or Sign in.
Decide from the page context whether a control opens or advances an application
form versus whether it finalizes an already completed application. Do not assume
button labels are universal; observe the page and use the available controls.

You must stop immediately and leave the browser open when any of these occur:
- CAPTCHA, bot check, OTP, MFA, security question, or email/SMS verification
- Login is required and no active session is available
- A required answer is absent from the profile or requires personal judgment
- A legal attestation, EEO disclosure, consent, signature, assessment, payment,
  background-check authorization, or similar acknowledgement is required
- The final review page or a control that finalizes, submits, sends, confirms,
  or finishes an already completed application is reached

Never click a control that finalizes an application. Browser safeguards block form
submission independently. Report what was completed and the precise reason you
stopped so the human can review and take the next step.
""".strip()


def build_navigation_only_task(application_url: str) -> str:
    """Build a bounded fallback task that never receives candidate profile data."""
    return f"""
Open this career-site application URL: {application_url}

Navigate only to the first application form using a page-specific non-final
entry control, such as Apply now, Start application, Continue, or Next.
Do not request profile data, upload files, fill fields, accept consent, create
an account, solve a CAPTCHA, or click any final/submission control. Stop as
soon as the first application form is visible, or when human review is needed.
Report the exact observed blocker or form state.
""".strip()