"""Retriever Agent (spec §17): produces GroundedContext only, never drafts a response."""

from logging_config import get_logger, trace
from retrieval.hybrid import extract_all_identifiers, retrieve
from schemas.intake import IntakeRequest
from schemas.retrieval import GroundedContext

logger = get_logger("nexus.agents.retriever")


@trace(logger)
def retrieve_grounded_context(intake: IntakeRequest) -> GroundedContext:
    identifiers = extract_all_identifiers(intake.customer_stated_intent)
    return retrieve(intake, identifiers)
