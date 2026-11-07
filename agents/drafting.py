"""Drafting Agent (spec §18): drafts only from GroundedContext, never invents facts."""

import guardrails
from agents import anthropic_client, degradation, prompts
from config import model_config
from logging_config import get_logger, trace
from schemas.intake import IntakeRequest
from schemas.retrieval import GroundedContext

logger = get_logger("nexus.agents.drafting")

_SYSTEM_PROMPT = (
    "You are the Nexus Drafting Agent for Meridian Financial Group. "
    "Draft a short, professional customer-facing response using ONLY the facts provided "
    "in the grounded context below. If a fact is not present in the grounded context, "
    "do not state it. Never reveal unmasked account/policy/portfolio identifiers; refer to "
    "them only in their masked form (e.g. ****0077) exactly as given. "
    "If structured and unstructured sources conflict, follow the structured (authoritative) value "
    "and do not silently ignore the conflict if asked about it. "
    "Be internally consistent: if the grounded context makes the answer conditional, do not open "
    "with an unqualified yes or no. Do not promise actions, call-backs, confirmations or contact "
    "channels that the grounded context does not state. "
    "Reply with the customer-facing text only, as plain text without Markdown formatting "
    "(no bold markers, headings or tables), no preamble."
)


def _summarize_context(context: GroundedContext) -> str:
    lines = ["Structured facts (authoritative):"]
    for fact in context.structured_facts:
        lines.append(f"- [{fact.source}] {fact.field} = {fact.value}")
    lines.append("Unstructured knowledge-base facts:")
    for fact in context.unstructured_facts:
        lines.append(f"- [{fact.document_id}] {fact.excerpt[:200]}")
    if context.conflicts:
        lines.append("Known conflicts (structured data wins):")
        for conflict in context.conflicts:
            lines.append(
                f"- {conflict.topic}: structured={conflict.structured_value}, "
                f"unstructured={conflict.unstructured_value} -> resolved via {conflict.resolution}"
            )
    return "\n".join(lines)


_CUSTOMER_SAFE_FIELD_LABELS = {
    "account_type": "Account type",
    "status": "Status",
    "currency": "Currency",
    "masked_account_id": "Account",
    "masked_policy_id": "Policy",
    "policy_type": "Policy type",
    "premium_amount": "Premium amount",
    "premium_frequency": "Premium frequency",
    "payment_terms_summary": "Payment terms",
    "covered": "Auto-debit covered under policy terms",
    "auto_debit_restricted": "Auto-debit restricted on this account",
    "masked_portfolio_id": "Portfolio",
    "risk_profile": "Risk profile",
    "masked_claim_id": "Claim",
    "recent_transaction": "Recent transaction",
    "product_name": "Product",
    "risk_level": "Product risk level",
    "minimum_investment": "Minimum investment",
    "suitable": "Suitable for your risk profile",
    "filed_at": "Filed on",
}


def _display_value(value: str) -> str:
    """Render stored flags in customer-readable form rather than raw 0/1 or True/False."""
    return {"0": "No", "1": "Yes", "True": "Yes", "False": "No"}.get(value, value)


def _rule_based_draft(intake: IntakeRequest, context: GroundedContext) -> str:
    """Deterministic customer-safe draft used when no LLM is configured.

    Only renders whitelisted, customer-appropriate fields: internal source/server names and
    raw knowledge-base chunks must not reach the customer.
    """
    if not context.structured_facts and not context.unstructured_facts:
        return (
            "Thank you for reaching out. We were unable to retrieve enough verified information "
            "to answer this request right now. A specialist will follow up with you."
        )

    lines = [
        "Thank you for your message. Here is what we can confirm from your records:"
    ]
    # De-duplicated on (field, value): two accounts named in one request both appear, while the
    # same value reported by two sources is shown once.
    seen: set[tuple[str, str]] = set()
    for fact in context.structured_facts:
        label = _CUSTOMER_SAFE_FIELD_LABELS.get(fact.field)
        if label is None or (fact.field, fact.value) in seen:
            continue
        seen.add((fact.field, fact.value))
        lines.append(f"- {label}: {_display_value(fact.value)}")

    if len(lines) == 1:
        lines.append(
            "- We located your records but have no further details to share here."
        )

    if context.conflicts:
        lines.append(
            "\nPlease note: our published guidance and your account's current settings differ "
            "on this point. Your account's live settings take precedence, and we have flagged "
            "this for review."
        )

    lines.append(
        "\nIf you have further questions, a Meridian representative can assist further."
    )
    return "\n".join(lines)


@trace(logger, redact_return=True)  # the draft may quote the customer's free text
def draft_response(intake: IntakeRequest, context: GroundedContext) -> str:
    # Nothing grounded -> nothing for a model to draft from. The fixed "unable to retrieve" text is
    # used instead of an LLM call, so an empty context can never produce invented facts (and the
    # wording is deterministic for validation; it was not in live-LLM runs).
    if not context.structured_facts and not context.unstructured_facts:
        return _rule_based_draft(intake, context)

    context_summary = _summarize_context(context)

    if anthropic_client.is_configured():
        try:
            # The customer's own text may contain raw identifiers; mask them so the model can
            # only ever echo the masked form (the output filter would otherwise block the draft).
            masked_intent = guardrails.scrub_identifiers(intake.customer_stated_intent)
            context_summary = guardrails.scrub_identifiers(context_summary)
            user_prompt = prompts.render(
                "response_drafting_prompt",
                {
                    "classified_intent": f"{intake.product_line}/{intake.request_type}: {masked_intent}",
                    "retrieved_context_summary": context_summary,
                },
                fallback=(
                    f"Customer intent: {masked_intent}\n"
                    f"Product line: {intake.product_line}\n\n"
                    f"Grounded context:\n{context_summary}"
                ),
            )
            return anthropic_client.complete_text(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=model_config.DRAFTING_MODEL,
                max_tokens=model_config.DRAFTING_MAX_TOKENS,
                effort=model_config.DRAFTING_EFFORT,
            )
        except Exception as exc:
            logger.warning(
                "LLM drafting failed; using rule-based draft",
                extra={
                    "payload": {
                        "event": "LOG",
                        "degraded": "drafting",
                        "error": str(exc),
                    }
                },
            )
            degradation.record("drafting", "llm_failed_rule_based_fallback")

    return _rule_based_draft(intake, context)
