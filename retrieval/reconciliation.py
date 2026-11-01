"""Reconciliation: structured MCP data wins conflicts with unstructured KB data (spec §14).

Topic-driven: see RECONCILIATION_RULES."""

import re
from dataclasses import dataclass
from typing import Callable

import guardrails
from schemas.retrieval import (
    Conflict,
    GroundedContext,
    StructuredFact,
    UnstructuredFact,
)

# A knowledge-base document states the *general* rule. A conditional caveat such as
# "permitted provided the account is not flagged as restricted" is still permissive, so that
# negated condition is neutralized before keyword matching -- otherwise the bare word
# "restricted" would be misread as a categorical prohibition.
#
# Deliberately limited to "not ... restricted": a negation like "auto-debit is not enabled" is a
# categorical prohibition, not a caveat, and must survive to be matched as restrictive.
_NEGATED_CONDITION_RE = re.compile(
    r"\bnot\s+(?:be\s+|being\s+|flagged\s+as\s+|marked\s+as\s+)?restricted\b",
    re.IGNORECASE,
)

_AUTO_DEBIT_KEYWORDS_PERMITTED = (
    "permitted",
    "allowed",
    "may pay premiums via",
    "supports",
    "is available",
)
_AUTO_DEBIT_KEYWORDS_RESTRICTED = (
    "restricted",
    "not enabled",
    "declined",
    "manual payment only",
    "manual payment methods only",
    "is not available",
)

# Product types a knowledge-base chunk may be specifically about. A chunk that is only about a
# different product type than the customer's live policy is not evidence about their policy.
_PRODUCT_TYPE_MARKERS = {
    "life_insurance": ("life insurance",),
    "auto_insurance": ("auto insurance", "vehicle insurance"),
    "home_insurance": ("home insurance", "homeowner"),
}


def _text_suggests_auto_debit_permitted(text: str) -> bool | None:
    """What the general guidance says about auto-debit: True/False, or None if unclear."""
    lowered = _NEGATED_CONDITION_RE.sub(" ", (text or "").lower())

    # A categorical prohibition outranks permissive wording elsewhere in the same chunk.
    if "manual payment" in lowered and "only" in lowered:
        return False

    permitted = any(kw in lowered for kw in _AUTO_DEBIT_KEYWORDS_PERMITTED)
    restricted = any(kw in lowered for kw in _AUTO_DEBIT_KEYWORDS_RESTRICTED)
    if permitted and not restricted:
        return True
    if restricted and not permitted:
        return False
    return None


def _chunk_is_product_specific(text: str) -> bool:
    """True when the chunk names at least one specific insurance product type."""
    lowered = text.lower()
    return any(
        marker in lowered
        for markers in _PRODUCT_TYPE_MARKERS.values()
        for marker in markers
    )


def _chunk_is_about_other_product(text: str, policy_type: str | None) -> bool:
    """True when the chunk only discusses a product type other than the customer's policy."""
    if not policy_type:
        return False

    lowered = text.lower()
    own_markers = _PRODUCT_TYPE_MARKERS.get(policy_type, ())
    if any(marker in lowered for marker in own_markers):
        return False

    for other_type, markers in _PRODUCT_TYPE_MARKERS.items():
        if other_type == policy_type:
            continue
        if any(marker in lowered for marker in markers):
            return True
    return False


# ---------------------------------------------------------------------------------------------
# Topic-driven reconciliation (spec §14, scorecard fix #5). Each rule names a topic, how to read
# the authoritative value from structured MCP facts, and how to read the general claim a KB chunk
# makes about the same topic. Whenever both exist and disagree, the structured value wins and the
# conflict is retained. Adding a topic is adding a row -- the auto-debit rule is one row of four.
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ReconciliationRule:
    topic: str
    structured_value: Callable[[list[StructuredFact]], object | None]
    kb_value: Callable[[str, list[StructuredFact]], object | None]
    chunk_applies: Callable[[str, list[StructuredFact]], bool] = (
        lambda text, facts: True
    )


def _fact(facts: list[StructuredFact], name: str) -> str | None:
    return next((f.value for f in facts if f.field == name), None)


def _truthy(value: str | None) -> bool | None:
    if value is None or value in ("None", ""):
        return None
    return value in ("1", "True", "true", "yes")


# --- auto-debit permission (the showcase / Stress Test 2 topic) -------------------------------
_AUTO_DEBIT_FIELDS = ("auto_debit_allowed", "auto_debit_restricted", "covered")


def _auto_debit_structured(facts: list[StructuredFact]) -> bool | None:
    return _structured_permits_auto_debit(
        [f for f in facts if f.field in _AUTO_DEBIT_FIELDS]
    )


def _auto_debit_chunk_applies(text: str, facts: list[StructuredFact]) -> bool:
    policy_type = _fact(facts, "policy_type")
    if _chunk_is_about_other_product(text, policy_type):
        return False
    if policy_type is None and _chunk_is_product_specific(text):
        # No policy is in the grounded context (e.g. the customer named no policy id), so a rule
        # that applies to one insurance product type is not evidence for or against this
        # customer's account-level auto-debit eligibility.
        return False
    return True


# --- investment product suitability for the customer's risk profile ---------------------------
_NOT_SUITABLE_RE = re.compile(
    r"\b(not (suitable|suited|appropriate|recommended)|unsuitable|inappropriate)\b",
    re.I,
)
_SUITABLE_RE = re.compile(r"\b(suitable|suited|appropriate|recommended)\b", re.I)


def _mentions_product(text: str, facts: list[StructuredFact]) -> bool:
    name = _fact(facts, "product_name")
    return bool(name) and name.lower() in text.lower()


def _suitability_kb(text: str, facts: list[StructuredFact]) -> bool | None:
    """What the product document says about suitability for *this* customer's risk profile."""
    profile = (_fact(facts, "risk_profile") or "").lower()
    if not profile:
        return None
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not re.search(rf"\b{re.escape(profile)}\b", sentence, re.I):
            continue
        if _NOT_SUITABLE_RE.search(sentence):
            return False
        if _SUITABLE_RE.search(sentence):
            return True
    return None


# --- product minimum investment ---------------------------------------------------------------
_MIN_INVESTMENT_RE = re.compile(
    r"minimum (?:initial )?investment (?:is|of)\s*(?:usd\s*)?\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    re.I,
)


def _min_investment_structured(facts: list[StructuredFact]) -> float | None:
    value = _fact(facts, "minimum_investment")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _min_investment_kb(text: str, facts: list[StructuredFact]) -> float | None:
    match = _MIN_INVESTMENT_RE.search(text)
    return float(match.group(1).replace(",", "")) if match else None


# --- premium frequency (categorical "... only" statements in policy documents) ----------------
_FREQUENCIES = ("monthly", "quarterly", "annually", "annual", "semi-annually")
_FREQ_ONLY_RE = re.compile(
    r"\b(billed|payable|due|paid)\s+(monthly|quarterly|annually|annual|semi-annually)\s+only\b",
    re.I,
)


def _frequency_structured(facts: list[StructuredFact]) -> str | None:
    value = (_fact(facts, "premium_frequency") or "").lower()
    return "annually" if value == "annual" else (value or None)


def _frequency_kb(text: str, facts: list[StructuredFact]) -> str | None:
    match = _FREQ_ONLY_RE.search(text)
    if not match:
        return None
    value = match.group(2).lower()
    return "annually" if value == "annual" else value


def _frequency_chunk_applies(text: str, facts: list[StructuredFact]) -> bool:
    policy_type = _fact(facts, "policy_type")
    return (
        bool(policy_type)
        and not _chunk_is_about_other_product(text, policy_type)
        and _chunk_is_product_specific(text)
    )


RECONCILIATION_RULES: tuple[ReconciliationRule, ...] = (
    ReconciliationRule(
        topic="auto_debit_permission",
        structured_value=_auto_debit_structured,
        kb_value=lambda text, facts: _text_suggests_auto_debit_permitted(text),
        chunk_applies=_auto_debit_chunk_applies,
    ),
    ReconciliationRule(
        topic="product_suitability",
        structured_value=lambda facts: _truthy(_fact(facts, "suitable")),
        kb_value=_suitability_kb,
        chunk_applies=_mentions_product,
    ),
    ReconciliationRule(
        topic="minimum_investment",
        structured_value=_min_investment_structured,
        kb_value=_min_investment_kb,
        chunk_applies=_mentions_product,
    ),
    ReconciliationRule(
        topic="premium_frequency",
        structured_value=_frequency_structured,
        kb_value=_frequency_kb,
        chunk_applies=_frequency_chunk_applies,
    ),
)


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 0.005
        except (TypeError, ValueError):
            return False
    return a == b


def reconcile(
    structured_facts: list[StructuredFact],
    unstructured_hits: list[dict],
    topics: list[str] | None = None,
) -> GroundedContext:
    """Merge structured (MCP) and unstructured (Chroma) retrieval into a single GroundedContext.

    Every rule in RECONCILIATION_RULES whose structured value is present is checked against each
    knowledge-base chunk the rule applies to. On disagreement the structured fact is
    authoritative and the conflict is retained rather than silently discarded (spec §14,
    Invariant 3). `topics` (scenario tags derived from the request) is recorded for traceability;
    rules fire on the data present, so a topic the customer did not name can still surface a
    conflict that matters to the answer.
    """
    unstructured_facts = []
    for hit in unstructured_hits:
        # Chroma may return None metadata for a chunk; treat it as empty rather than crashing.
        metadata = hit.get("metadata") or {}
        unstructured_facts.append(
            UnstructuredFact(
                document_id=metadata.get("document_id")
                or hit.get("chunk_id")
                or "unknown",
                section=metadata.get("section"),
                excerpt=guardrails.sanitize_retrieved_text(hit.get("text") or ""),
            )
        )

    conflicts: list[Conflict] = []
    for rule in RECONCILIATION_RULES:
        structured_value = rule.structured_value(structured_facts)
        if structured_value is None:
            continue
        for uf in unstructured_facts:
            if not rule.chunk_applies(uf.excerpt, structured_facts):
                continue
            kb_value = rule.kb_value(uf.excerpt, structured_facts)
            if kb_value is None or _same(kb_value, structured_value):
                continue
            conflict = Conflict(
                topic=rule.topic,
                structured_value=str(structured_value),
                unstructured_value=str(kb_value),
                resolution="structured_authoritative",
            )
            # Several chunks can carry the same contradiction; report it once.
            if conflict not in conflicts:
                conflicts.append(conflict)

    sources = list({f.source for f in structured_facts}) + list(
        {uf.document_id for uf in unstructured_facts}
    )

    return GroundedContext(
        structured_facts=structured_facts,
        unstructured_facts=unstructured_facts,
        conflicts=conflicts,
        sources=sources,
        retrieval_status="ok",
    )


def _structured_permits_auto_debit(facts: list[StructuredFact]) -> bool | None:
    """Conservative: auto-debit is permitted only if every authoritative signal allows it.

    A restricted account and a permissive policy clause must not cancel out into "permitted".
    """
    truthy = ("1", "True", "true")
    signals: list[bool] = []
    for fact in facts:
        if fact.field == "auto_debit_restricted":
            signals.append(fact.value not in truthy)
        elif fact.field in ("auto_debit_allowed", "covered"):
            signals.append(fact.value in truthy)

    if not signals:
        return None
    return all(signals)
