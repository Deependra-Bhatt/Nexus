"""Banking MCP server (spec §21.1). Exposes 3 tools over stdio via FastMCP."""

from fastmcp import FastMCP

import guardrails
from errors import InvalidIdentifierError, MCPToolError
from logging_config import get_logger, trace
from mcp_servers.banking_server.repository import BankingRepository

logger = get_logger("nexus.banking_server")
repo = BankingRepository()

mcp = FastMCP("banking_server")


def _load_owned_account(account_id: str, customer_id: str) -> dict:
    """Validate the id, load the account and enforce that it belongs to the (mandatory) caller
    customer_id (spec §51 rule 5)."""
    guardrails.validate_identifier(account_id, "account_id")
    account = repo.get_account(account_id)
    guardrails.assert_record_owner(account, customer_id, "account")
    if account is None:
        raise InvalidIdentifierError(f"Account not found: {account_id!r}")
    return account


@mcp.tool()
@trace(logger)
def get_account_summary(
    account_id: str, customer_id: str, caller_scope: str = "retrieval"
) -> dict:
    """Look up an account by id, returning field-minimized, identifier-masked data.
    customer_id is required and the account must belong to that customer. The returned field
    scope is decided server-side (guardrails.resolve_caller_scope), not by the caller.
    """
    account = _load_owned_account(account_id, customer_id)
    account["masked_account_id"] = guardrails.mask_sensitive_identifier(
        account["account_id"]
    )
    return guardrails.minimize_account_fields(
        account, caller_scope=guardrails.resolve_caller_scope(caller_scope)
    )


@mcp.tool()
@trace(logger)
def get_transaction_history(
    account_id: str, customer_id: str, limit: int = 10
) -> list[dict]:
    """Return the most recent transactions for an account, most recent first.
    customer_id is required and the account must belong to that customer."""
    guardrails.validate_identifier(account_id, "account_id")
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
        or limit > 100
    ):
        raise MCPToolError("limit must be between 1 and 100")
    _load_owned_account(account_id, customer_id)

    transactions = repo.get_transactions(account_id, limit=limit)
    return [
        {
            "transaction_id": txn["transaction_id"],
            "posted_at": txn["posted_at"],
            "description": txn["description"],
            "amount": txn["amount"],
            "direction": txn["direction"],
        }
        for txn in transactions
    ]


@mcp.tool()
@trace(logger)
def get_linked_accounts(customer_id: str) -> list[dict]:
    """Return accounts linked to a customer, with the same minimization policy."""
    guardrails.validate_identifier(customer_id, "customer_id")
    accounts = repo.get_accounts_for_customer(customer_id)
    results = []
    for account in accounts:
        account["masked_account_id"] = guardrails.mask_sensitive_identifier(
            account["account_id"]
        )
        results.append(
            guardrails.minimize_account_fields(account, caller_scope="retrieval")
        )
    return results


@mcp.resource("account://{account_id}/summary")
@trace(logger)
def account_summary_resource(account_id: str) -> dict:
    """Masked, minimized account summary resource (spec §25). Not for bulk data."""
    guardrails.validate_identifier(account_id, "account_id")
    account = repo.get_account(account_id)
    if account is None:
        raise InvalidIdentifierError(f"Account not found: {account_id!r}")

    account["masked_account_id"] = guardrails.mask_sensitive_identifier(
        account["account_id"]
    )
    return guardrails.minimize_account_fields(account, caller_scope="retrieval")


if __name__ == "__main__":
    from mcp_servers.startup import announce_startup

    announce_startup(logger, "banking_server", mcp)
    mcp.run()
