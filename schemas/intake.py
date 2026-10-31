"""Structured intake schema (spec §9)."""

import re
import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Same strictness as guardrails.validate_identifier: ASCII digits only and a full match, so a
# trailing newline or full-width digits cannot pass the intake gate.
_CUSTOMER_ID_RE = re.compile(r"CUS-[0-9]{5}")


class IntakeRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: f"REQ-{uuid.uuid4().hex[:10]}")
    customer_id: str
    product_line: Literal["banking", "insurance", "wealth", "cross_product"]
    request_type: str
    urgency: Literal["routine", "high", "critical"]
    customer_stated_intent: str
    escalation_category: Literal[
        "none",
        "financial_hardship",
        "safeguarding_concern",
        "compliance_override_request",
    ]

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id(cls, value: str) -> str:
        if not isinstance(value, str) or not _CUSTOMER_ID_RE.fullmatch(value):
            raise ValueError(f"Invalid customer_id format: {value!r}")
        return value

    @field_validator("customer_stated_intent")
    @classmethod
    def validate_intent_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("customer_stated_intent must not be empty")
        return value
