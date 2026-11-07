"""Escalation Agent (spec §20): deterministic, delegates to run_escalation_check MCP tool."""

import re

from config import settings
from errors import EscalationError, MCPServerUnavailableError, MCPToolError
from logging_config import get_logger, trace
from mcp_servers.client import call_tool_sync
from schemas.intake import IntakeRequest
from schemas.validation import ValidationResult

logger = get_logger("nexus.agents.escalation")

# Layer 3 of hard-escalation defence: emotional-distress language that matched none of the three
# hard categories. It does not invent a category; it caps the effective confidence below the MCP
# threshold so the same deterministic run_escalation_check routes the request to a human
# ("low_confidence"). This generalises to paraphrases no pattern anticipated, at the cost of an
# occasional unnecessary handoff -- the safe direction for a financial concierge.
_DISTRESS_RE = re.compile(
    r"\b(scared|afraid|frightened|terrified|desperate|hopeless|helpless|panic\w*|crying|in tears|"
    r"threat\w*|harass\w*|intimidat\w*|don'?t know what to do|wits'? end|can'?t cope|can'?t take (it|this)|"
    r"overwhelm\w*|worried sick|at breaking point|breaking down|falling apart|"
    r"can'?t think straight|losing my mind|nowhere to turn|no one to turn to)\b",
    re.I,
)


def distress_signal(text: str) -> bool:
    """True when the customer's own words carry emotional-distress language."""
    normalised = (text or "").replace("\u2019", "'").replace("\u2018", "'")
    return bool(_DISTRESS_RE.search(normalised))


@trace(logger)
def check_escalation(
    intake: IntakeRequest,
    validation: ValidationResult,
    security_review: str | None = None,
    security_flags: list[str] | None = None,
) -> dict:
    """Never substitutes LLM judgement for the deterministic MCP decision (Invariant 5).

    A draft that failed validation (passed=False) must not be delivered no matter how high the
    reported confidence is (spec B16), so its effective confidence for the deterministic check
    is 0.0, which routes it to human review through the same MCP decision path.

    `security_review` (e.g. "prompt_injection_suspected") is set for quarantined requests. A hard
    customer category detected in the same message still wins -- an injection phrase appended to a
    hardship statement must not divert it away from the hardship team -- otherwise the security
    category is sent, which the MCP tool always escalates.
    """
    effective_confidence = validation.confidence if validation.passed else 0.0
    if not validation.passed:
        logger.debug(
            "Validation failed; escalating via zero effective confidence",
            extra={
                "payload": {
                    "event": "LOG",
                    "reported_confidence": validation.confidence,
                    "issues": validation.issues,
                }
            },
        )
    if intake.escalation_category == "none" and distress_signal(
        intake.customer_stated_intent
    ):
        capped = min(effective_confidence, settings.DISTRESS_CONFIDENCE_CAP)
        if capped < effective_confidence:
            logger.info(
                "Distress language detected; confidence capped so a human reviews the request",
                extra={
                    "payload": {
                        "event": "LOG",
                        "reported_confidence": effective_confidence,
                        "cap": capped,
                    }
                },
            )
        effective_confidence = capped
    request_category = intake.escalation_category
    if security_review and request_category not in settings.HARD_ESCALATION_CATEGORIES:
        if security_review not in settings.SECURITY_REVIEW_CATEGORIES:
            raise EscalationError(
                f"Unknown security review category: {security_review!r}"
            )
        request_category = security_review
    try:
        decision = call_tool_sync(
            "concierge_ops",
            "run_escalation_check",
            {
                "customer_id": intake.customer_id,
                "request_category": request_category,
                "draft_confidence": effective_confidence,
                "request_id": intake.request_id,
                # Triage context stored on the escalation case (never used for the decision).
                "urgency": intake.urgency,
                "product_line": intake.product_line,
                "request_type": intake.request_type,
                "security_flags": list(security_flags or []),
            },
        )
    except (MCPServerUnavailableError, MCPToolError):
        raise
    except Exception as exc:
        raise EscalationError(f"Escalation check failed: {exc}") from exc

    if not isinstance(decision, dict) or "escalate" not in decision:
        raise EscalationError(
            f"Escalation check returned an unexpected payload: {decision!r}"
        )
    return decision
