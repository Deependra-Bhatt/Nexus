"""Wealth MCP server (spec §23). Exposes 3 tools over stdio via FastMCP."""

from fastmcp import FastMCP

import guardrails
from errors import InvalidIdentifierError, InvalidInputError
from logging_config import get_logger, trace
from mcp_servers.wealth_server.repository import WealthRepository

logger = get_logger("nexus.wealth_server")
repo = WealthRepository()

mcp = FastMCP("wealth_server")

_RISK_ORDER = {"low": 0, "moderate": 1, "high": 2}


@mcp.tool()
@trace(logger)
def get_portfolio_summary(
    portfolio_id: str, customer_id: str, caller_scope: str = "retrieval"
) -> dict:
    """Portfolio lookup with field minimization.
    customer_id is required and the portfolio must belong to that customer. The returned field
    scope is decided server-side (guardrails.resolve_caller_scope), not by the caller.
    """
    guardrails.validate_identifier(portfolio_id, "portfolio_id")
    portfolio = repo.get_portfolio(portfolio_id)
    guardrails.assert_record_owner(portfolio, customer_id, "portfolio")
    if portfolio is None:
        raise InvalidIdentifierError(f"Portfolio not found: {portfolio_id!r}")

    portfolio["masked_portfolio_id"] = guardrails.mask_sensitive_identifier(
        portfolio["portfolio_id"]
    )
    return guardrails.minimize_portfolio_fields(
        portfolio, caller_scope=guardrails.resolve_caller_scope(caller_scope)
    )


@mcp.tool()
@trace(logger)
def get_investment_product_details(product_id: str) -> dict:
    """Single investment-product lookup."""
    guardrails.validate_identifier(product_id, "product_id")
    product = repo.get_investment_product(product_id)
    if product is None:
        raise InvalidIdentifierError(f"Investment product not found: {product_id!r}")

    return {
        "product_id": product["product_id"],
        "product_name": product["product_name"],
        "risk_level": product["risk_level"],
        "minimum_investment": product["minimum_investment"],
        "description": product["description"],
    }


@mcp.tool()
@trace(logger)
def check_product_suitability(product_id: str, risk_profile: str) -> dict:
    """Suitability check only. No write capability, no additional customer data beyond risk_profile."""
    guardrails.validate_identifier(product_id, "product_id")
    product = repo.get_investment_product(product_id)
    if product is None:
        raise InvalidIdentifierError(f"Investment product not found: {product_id!r}")

    if risk_profile not in _RISK_ORDER:
        # Typed error (spec §36) rather than a bare ValueError.
        raise InvalidInputError(
            f"Unknown risk_profile: {risk_profile!r}; expected one of {sorted(_RISK_ORDER)}"
        )
    if product["risk_level"] not in _RISK_ORDER:
        raise InvalidInputError(
            f"Product has an unknown risk_level: {product['risk_level']!r}"
        )

    suitable = _RISK_ORDER[risk_profile] >= _RISK_ORDER[product["risk_level"]]
    return {
        "product_id": product_id,
        "risk_profile": risk_profile,
        "product_risk_level": product["risk_level"],
        "suitable": suitable,
    }


if __name__ == "__main__":
    from mcp_servers.startup import announce_startup

    announce_startup(logger, "wealth_server", mcp)
    mcp.run()
