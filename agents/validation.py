"""Validation Agent (spec §19): checks draft against GroundedContext, scores confidence."""

import re

import guardrails
from agents import anthropic_client, degradation
from config import model_config
from logging_config import get_logger, trace
from schemas.intake import IntakeRequest
from schemas.retrieval import GroundedContext
from schemas.validation import ValidationResult

logger = get_logger("nexus.agents.validation")

_SYSTEM_PROMPT = (
    "You are the Nexus Validation Agent. Compare the draft response against the grounded context. "
    "Check factual accuracy, tone, compliance, and whether any claim is unsupported by the grounded "
    "context. Return a JSON object: "
    '{"passed": bool, "confidence": float between 0 and 1, "issues": [strings], "conflict_present": bool}. '
    "Set passed=false if the draft states anything not supported by the grounded context."
)

_VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "confidence": {"type": "number"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "conflict_present": {"type": "boolean"},
    },
    "required": ["passed", "confidence", "issues", "conflict_present"],
    "additionalProperties": False,
}


# Deterministic grounding check (scorecard 4.6 / 4.4). Independent of any model: every monetary
# figure, percentage and masked identifier the draft states must occur in the grounded context.
# It runs in rule-based AND LLM mode, so a validator that shares the drafter's provider cannot
# wave through an invented premium, balance or account.
_MONEY_RE = re.compile(
    r"(?:[$€£₹]|\b(?:usd|inr|eur|gbp)\s?)\s?([0-9][0-9,]*(?:\.[0-9]+)?)|\b([0-9][0-9,]*\.[0-9]{1,2})\b",
    re.I,
)
_PERCENT_RE = re.compile(r"\b([0-9]+(?:\.[0-9]+)?)\s?%")
_MASKED_RE = re.compile(r"\*{4}[A-Za-z0-9-]{2,}")
_NUMBER_RE = re.compile(r"[0-9][0-9,]*(?:\.[0-9]+)?")


def _numbers_in(text: str) -> set[float]:
    values = set()
    for raw in _NUMBER_RE.findall(text or ""):
        try:
            values.add(round(float(raw.replace(",", "")), 2))
        except ValueError:
            continue
    return values


def grounding_issues(draft: str, context: GroundedContext) -> list[str]:
    """Return one issue per figure / masked id in the draft that the grounded context lacks."""
    corpus = " ".join(
        [f"{f.field} {f.value}" for f in context.structured_facts]
        + [f.excerpt for f in context.unstructured_facts]
        + [f"{c.structured_value} {c.unstructured_value}" for c in context.conflicts]
    )
    known_numbers = _numbers_in(corpus)
    issues: list[str] = []
    for match in _MONEY_RE.finditer(draft or ""):
        raw = (match.group(1) or match.group(2) or "").replace(",", "")
        try:
            value = round(float(raw), 2)
        except ValueError:
            continue
        if value not in known_numbers:
            issues.append(f"Unsupported figure in draft: {match.group(0).strip()}")
    for match in _PERCENT_RE.finditer(draft or ""):
        if round(float(match.group(1)), 2) not in known_numbers:
            issues.append(f"Unsupported percentage in draft: {match.group(0).strip()}")
    for match in _MASKED_RE.finditer(draft or ""):
        if match.group(0) not in corpus:
            issues.append(
                f"Masked identifier not in grounded context: {match.group(0)}"
            )
    return issues


def _deterministic_score(
    intake: IntakeRequest, draft: str, context: GroundedContext
) -> ValidationResult:
    issues: list[str] = []
    conflict_present = bool(context.conflicts)

    if not draft or not draft.strip():
        return ValidationResult(
            passed=False,
            confidence=0.0,
            issues=["Draft is empty"],
            conflict_present=conflict_present,
        )

    has_facts = bool(context.structured_facts or context.unstructured_facts)
    if not has_facts:
        issues.append(
            "No structured or unstructured facts available to ground the draft"
        )
        confidence = 0.2
    elif conflict_present:
        issues.append(
            "Retrieval conflict present between structured and unstructured sources"
        )
        confidence = 0.6
    elif context.structured_facts:
        confidence = 0.92
    else:
        confidence = 0.55

    passed = bool(draft.strip())
    unsupported = grounding_issues(draft, context)
    if unsupported:
        # A stated figure/identifier the context cannot back is a grounding failure, not a style
        # issue: the draft is not delivered (check_escalation treats passed=False as 0.0).
        issues.extend(unsupported)
        passed = False
    return ValidationResult(
        passed=passed,
        confidence=confidence,
        issues=issues,
        conflict_present=conflict_present,
    )


@trace(
    logger, redact_return=True
)  # draft may quote customer's own words back to them (TC-197)
def validate_draft(
    intake: IntakeRequest, draft: str, context: GroundedContext
) -> ValidationResult:
    baseline = _deterministic_score(intake, draft, context)

    if anthropic_client.is_configured():
        try:
            context_summary = (
                f"Structured facts: {[f.model_dump() for f in context.structured_facts]}\n"
                f"Unstructured facts: {[f.model_dump() for f in context.unstructured_facts]}\n"
                f"Conflicts: {[c.model_dump() for c in context.conflicts]}"
            )
            raw = anthropic_client.complete_json(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=(
                    f"Draft:\n{guardrails.scrub_identifiers(draft)}\n\n"
                    f"Grounded context:\n{guardrails.scrub_identifiers(context_summary)}"
                ),
                model=model_config.VALIDATION_MODEL,
                max_tokens=model_config.VALIDATION_MAX_TOKENS,
                effort=model_config.VALIDATION_EFFORT,
                schema=_VALIDATION_SCHEMA,
            )
            llm_result = ValidationResult(**raw)
            # A retrieval conflict detected deterministically is never un-flagged by the model.
            llm_result.conflict_present = (
                llm_result.conflict_present or baseline.conflict_present
            )
            # The model may LOWER confidence but never raise it above the deterministic grounding
            # score: with no structured facts (0.55), none at all (0.2) or a retained conflict (0.6),
            # a confident-sounding validator must not push the draft past the escalation threshold.
            llm_result.confidence = min(llm_result.confidence, baseline.confidence)
            # Nor can the model pass a draft that fails the deterministic grounding check.
            if not baseline.passed:
                llm_result.passed = False
                llm_result.issues = list(llm_result.issues) + [
                    i for i in baseline.issues if i not in llm_result.issues
                ]
            return llm_result
        except Exception as exc:
            logger.warning(
                "LLM validation failed; using deterministic validation score",
                extra={
                    "payload": {
                        "event": "LOG",
                        "degraded": "validation",
                        "error": str(exc),
                    }
                },
            )
            degradation.record("validation", "llm_failed_rule_based_fallback")

    return baseline
