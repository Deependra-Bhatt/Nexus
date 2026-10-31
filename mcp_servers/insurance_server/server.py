"""Insurance MCP server (spec §22). Exposes 3 tools over stdio via FastMCP."""

from fastmcp import FastMCP

import guardrails
from errors import InvalidIdentifierError
from logging_config import get_logger, trace
from mcp_servers.insurance_server.repository import InsuranceRepository

logger = get_logger("nexus.insurance_server")
repo = InsuranceRepository()

mcp = FastMCP("insurance_server")

_COVERAGE_SCENARIOS = {
    "auto_debit": lambda policy: bool(policy["auto_debit_allowed"]),
}


def _load_owned_policy(policy_id: str, customer_id: str) -> dict:
    guardrails.validate_identifier(policy_id, "policy_id")
    policy = repo.get_policy(policy_id)
    guardrails.assert_record_owner(policy, customer_id, "policy")
    if policy is None:
        raise InvalidIdentifierError(f"Policy not found: {policy_id!r}")
    return policy


@mcp.tool()
@trace(logger)
def get_policy_details(policy_id: str, customer_id: str) -> dict:
    """Return premiums, payment terms, status and other permitted policy fields.
    customer_id is required and the policy must belong to that customer."""
    policy = _load_owned_policy(policy_id, customer_id)
    return {
        "masked_policy_id": guardrails.mask_sensitive_identifier(policy["policy_id"]),
        "policy_type": policy["policy_type"],
        "status": policy["status"],
        "premium_amount": policy["premium_amount"],
        "premium_frequency": policy["premium_frequency"],
        "payment_terms_summary": policy["payment_terms_summary"],
    }


@mcp.tool()
@trace(logger)
def check_coverage_clause(policy_id: str, scenario_tag: str, customer_id: str) -> dict:
    """Deterministic coverage check for a named scenario (e.g. auto_debit).
    customer_id is required and the policy must belong to that customer."""
    policy = _load_owned_policy(policy_id, customer_id)

    checker = _COVERAGE_SCENARIOS.get(scenario_tag)
    if checker is None:
        return {
            "scenario_tag": scenario_tag,
            "covered": None,
            "reason": "Unknown scenario_tag",
        }

    covered = checker(policy)
    return {
        "scenario_tag": scenario_tag,
        "covered": covered,
        "reason": policy["payment_terms_summary"],
    }


@mcp.tool()
@trace(logger)
def get_claim_status(claim_id: str, customer_id: str) -> dict:
    """Return claim status only. No settlement authority.
    customer_id is required and the claim's policy must belong to that customer."""
    guardrails.validate_identifier(claim_id, "claim_id")
    claim = repo.get_claim(claim_id)
    guardrails.assert_record_owner(claim, customer_id, "claim")
    if claim is None:
        raise InvalidIdentifierError(f"Claim not found: {claim_id!r}")

    return {
        "masked_claim_id": guardrails.mask_sensitive_identifier(claim["claim_id"]),
        "status": claim["status"],
        "filed_at": claim["filed_at"],
    }


@mcp.resource("policy://{policy_id}/summary")
@trace(logger)
def policy_summary_resource(policy_id: str) -> dict:
    """Masked policy summary resource (spec §25). Not for bulk data."""
    guardrails.validate_identifier(policy_id, "policy_id")
    policy = repo.get_policy(policy_id)
    if policy is None:
        raise InvalidIdentifierError(f"Policy not found: {policy_id!r}")

    return {
        "masked_policy_id": guardrails.mask_sensitive_identifier(policy["policy_id"]),
        "policy_type": policy["policy_type"],
        "status": policy["status"],
        "payment_terms_summary": policy["payment_terms_summary"],
    }


if __name__ == "__main__":
    from mcp_servers.startup import announce_startup

    announce_startup(logger, "insurance_server", mcp)
    mcp.run()
