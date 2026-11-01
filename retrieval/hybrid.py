"""Hybrid retrieval: combines structured MCP data with unstructured Chroma data (spec §12)."""

import re

import progress
from config import settings
from errors import (
    InvalidIdentifierError,
    MCPServerUnavailableError,
    MCPToolError,
    RetrievalError,
    UnauthorizedFieldAccessError,
)
from logging_config import get_logger, trace
from mcp_servers.client import call_tool_sync
from retrieval.chroma_store import semantic_search
from retrieval.reconciliation import reconcile
from schemas.intake import IntakeRequest
from schemas.retrieval import GroundedContext, StructuredFact

logger = get_logger("nexus.hybrid_retrieval")

_ID_PATTERNS = {
    "account_id": re.compile(r"\bACC-[0-9]{5}\b"),
    "policy_id": re.compile(r"\bPOL-[A-Z]{2}-[0-9]{5}\b"),
    "portfolio_id": re.compile(r"\bPORT-[0-9]{5}\b"),
    "claim_id": re.compile(r"\bCLM-[0-9]{5}\b"),
    "product_id": re.compile(r"\bPROD-[A-Z0-9]{2}-[A-Z0-9]{2}\b"),
}


def extract_identifiers(text: str) -> dict[str, str]:
    """First identifier of each type (kept for callers that need a single value)."""
    found = {}
    for key, pattern in _ID_PATTERNS.items():
        match = pattern.search(text or "")
        if match:
            found[key] = match.group(0)
    return found


def extract_all_identifiers(
    text: str, limit: int | None = None
) -> dict[str, list[str]]:
    """Every distinct identifier of each type, in order of appearance (BUG-34).

    Capped at MAX_IDENTIFIERS_PER_TYPE per type, so a customer naming two accounts gets facts for
    both while one request can never fan out into an unbounded number of MCP calls.
    """
    limit = limit or settings.MAX_IDENTIFIERS_PER_TYPE
    found: dict[str, list[str]] = {}
    for key, pattern in _ID_PATTERNS.items():
        values: list[str] = []
        for match in pattern.finditer(text or ""):
            if match.group(0) not in values:
                values.append(match.group(0))
            if len(values) >= limit:
                break
        if values:
            found[key] = values
    return found


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [v for v in value if v]
    return [value]


# Coverage scenarios the insurance server can check deterministically, and the request wording
# that asks about each. Derived from the intake text instead of always asking for "auto_debit".
SCENARIO_TOPICS: dict[str, re.Pattern] = {
    "auto_debit": re.compile(
        r"\b(auto[- ]?debit|autopay|auto[- ]?pay|direct debit|automatic (debit|payment)s?|standing (order|instruction)|"
        r"pay (my )?premiums? (from|out of|via) (my )?(bank|account|savings|checking))\b",
        re.I,
    ),
}
_TRANSACTIONS_RE = re.compile(
    r"\b(transactions?|statement|recent (activity|payments?|debits?)|spending|charged)\b",
    re.I,
)
_SUITABILITY_RE = re.compile(
    r"\b(suitab\w*|right for me|good fit|appropriate for me|should i invest|can i invest)\b",
    re.I,
)
_TRANSACTION_FACT_LIMIT = 5


def scenario_tags_for(text: str) -> list[str]:
    """Coverage scenarios mentioned in the request (empty when the request asks about none)."""
    return [
        tag for tag, pattern in SCENARIO_TOPICS.items() if pattern.search(text or "")
    ]


def _facts(source: str, payload: dict) -> list[StructuredFact]:
    if not isinstance(payload, dict):
        raise RetrievalError(
            f"{source} returned an unexpected payload type: {type(payload).__name__}"
        )
    return [
        StructuredFact(source=source, field=name, value=str(value))
        for name, value in payload.items()
    ]


@trace(logger)
def retrieve_structured(
    intake: IntakeRequest, identifiers: dict
) -> list[StructuredFact]:
    """Call the relevant MCP server(s) based on product_line and known identifiers (spec §12.2).

    Every customer-linked lookup passes the intake's (server-trusted) customer_id so the owning
    MCP server can refuse records that belong to someone else (spec §51 rule 5). `identifiers`
    maps a type to one id or a list of ids (see extract_all_identifiers).
    """
    facts: list[StructuredFact] = []
    customer_id = intake.customer_id
    text = intake.customer_stated_intent
    line = intake.product_line

    account_ids = _as_list(identifiers.get("account_id"))
    if line in ("banking", "cross_product") and not account_ids:
        # No account_id was named in the request text; fall back to facts about the
        # customer's linked accounts (identifiers stay masked, per minimization policy).
        linked = call_tool_sync(
            "banking", "get_linked_accounts", {"customer_id": customer_id}
        )
        if not isinstance(linked, list):
            raise RetrievalError(
                f"banking_server returned an unexpected payload type: {type(linked).__name__}"
            )
        for account in linked:
            facts.extend(_facts("banking_server", account))

    if line in ("banking", "cross_product"):
        wants_transactions = bool(_TRANSACTIONS_RE.search(text or ""))
        for account_id in account_ids:
            summary = call_tool_sync(
                "banking",
                "get_account_summary",
                {"account_id": account_id, "customer_id": customer_id},
            )
            facts.extend(_facts("banking_server", summary))
            if wants_transactions:
                history = call_tool_sync(
                    "banking",
                    "get_transaction_history",
                    {
                        "account_id": account_id,
                        "customer_id": customer_id,
                        "limit": _TRANSACTION_FACT_LIMIT,
                    },
                )
                if not isinstance(history, list):
                    raise RetrievalError(
                        "banking_server returned an unexpected transaction payload"
                    )
                masked = (
                    summary.get("masked_account_id", "")
                    if isinstance(summary, dict)
                    else ""
                )
                for txn in history:
                    facts.append(
                        StructuredFact(
                            source="banking_server",
                            field="recent_transaction",
                            value=f"{masked} {str(txn.get('posted_at', ''))[:10]} {txn.get('description', '')} "
                            f"{txn.get('amount')} ({txn.get('direction', '')})".strip(),
                        )
                    )

    if line in ("insurance", "cross_product"):
        scenario_tags = scenario_tags_for(text)
        for policy_id in _as_list(identifiers.get("policy_id")):
            policy = call_tool_sync(
                "insurance",
                "get_policy_details",
                {"policy_id": policy_id, "customer_id": customer_id},
            )
            facts.extend(_facts("insurance_server", policy))
            for scenario_tag in scenario_tags:
                coverage = call_tool_sync(
                    "insurance",
                    "check_coverage_clause",
                    {
                        "policy_id": policy_id,
                        "scenario_tag": scenario_tag,
                        "customer_id": customer_id,
                    },
                )
                # Field name kept as "covered" for the auto_debit scenario (reconciliation rule and
                # drafting labels key on it); other scenarios are namespaced.
                field_name = (
                    "covered"
                    if scenario_tag == "auto_debit"
                    else f"covered_{scenario_tag}"
                )
                facts.append(
                    StructuredFact(
                        source="insurance_server",
                        field=field_name,
                        value=str(coverage.get("covered")),
                    )
                )

        for claim_id in _as_list(identifiers.get("claim_id")):
            claim = call_tool_sync(
                "insurance",
                "get_claim_status",
                {"claim_id": claim_id, "customer_id": customer_id},
            )
            facts.extend(_facts("insurance_server", claim))

    if line in ("wealth", "cross_product"):
        risk_profile = None
        for portfolio_id in _as_list(identifiers.get("portfolio_id")):
            portfolio = call_tool_sync(
                "wealth",
                "get_portfolio_summary",
                {"portfolio_id": portfolio_id, "customer_id": customer_id},
            )
            facts.extend(_facts("wealth_server", portfolio))
            if isinstance(portfolio, dict) and risk_profile is None:
                risk_profile = portfolio.get("risk_profile")

        for product_id in _as_list(identifiers.get("product_id")):
            product = call_tool_sync(
                "wealth", "get_investment_product_details", {"product_id": product_id}
            )
            facts.extend(_facts("wealth_server", product))
            # Suitability is decided by the wealth server against the customer's own risk profile,
            # so it is only asked when a portfolio (hence a risk profile) is in the request and the
            # customer is asking about suitability.
            if risk_profile and _SUITABILITY_RE.search(text or ""):
                suitability = call_tool_sync(
                    "wealth",
                    "check_product_suitability",
                    {"product_id": product_id, "risk_profile": risk_profile},
                )
                facts.append(
                    StructuredFact(
                        source="wealth_server",
                        field="suitable",
                        value=str(suitability.get("suitable")),
                    )
                )

    return facts


@trace(logger)
def retrieve(
    intake: IntakeRequest, identifiers: dict[str, str] | None = None
) -> GroundedContext:
    """Full hybrid retrieval + reconciliation pipeline (spec §12, §14).

    Raises MCPServerUnavailableError if a required backend is offline rather than
    silently substituting fabricated data (Invariant 8).
    """
    identifiers = identifiers or extract_all_identifiers(intake.customer_stated_intent)

    try:
        structured_facts = retrieve_structured(intake, identifiers)
    except (
        MCPServerUnavailableError,
        MCPToolError,
        UnauthorizedFieldAccessError,
        InvalidIdentifierError,
        RetrievalError,
    ):
        # Typed errors keep their meaning so the orchestrator can choose the right customer message.
        raise
    except Exception as exc:
        raise RetrievalError(f"Structured retrieval failed: {exc}") from exc

    progress.emit(
        "retrieval",
        "Structured retrieval via MCP",
        status="done",
        detail=f"{len(structured_facts)} authoritative fact(s) from "
        f"{', '.join(sorted({f.source for f in structured_facts})) or 'no server'}",
    )

    progress.emit(
        "retrieval", "Semantic search (Chroma knowledge base)", status="start"
    )
    try:
        unstructured_hits = semantic_search(
            intake.customer_stated_intent, product_line=intake.product_line
        )
    except Exception as exc:
        progress.emit(
            "retrieval",
            "Semantic search (Chroma knowledge base)",
            status="error",
            detail=type(exc).__name__,
        )
        raise RetrievalError(f"Unstructured retrieval failed: {exc}") from exc
    progress.emit(
        "retrieval",
        "Semantic search (Chroma knowledge base)",
        status="done",
        detail=f"{len(unstructured_hits)} relevant chunk(s)",
    )

    context = reconcile(
        structured_facts,
        unstructured_hits,
        topics=scenario_tags_for(intake.customer_stated_intent),
    )
    progress.emit(
        "retrieval",
        "Reconciliation (structured data wins on conflict)",
        status="done",
        detail=f"{len(context.conflicts)} conflict(s): "
        + (", ".join(c.topic for c in context.conflicts) or "none"),
    )
    return context
