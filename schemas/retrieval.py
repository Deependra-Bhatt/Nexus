"""Grounded context schema produced by hybrid retrieval + reconciliation (spec §14)."""

from pydantic import BaseModel, Field


class StructuredFact(BaseModel):
    source: str
    field: str
    value: str


class UnstructuredFact(BaseModel):
    document_id: str
    section: str | None = None
    excerpt: str


class Conflict(BaseModel):
    topic: str
    structured_value: str
    unstructured_value: str
    resolution: str = "structured_authoritative"


class GroundedContext(BaseModel):
    structured_facts: list[StructuredFact] = Field(default_factory=list)
    unstructured_facts: list[UnstructuredFact] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    retrieval_status: str = "ok"
