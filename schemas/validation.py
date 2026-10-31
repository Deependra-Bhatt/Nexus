"""Validation Agent result schema (spec §19)."""

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    conflict_present: bool = False
