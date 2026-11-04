"""In-process MCP tool dispatch: calls the real tool implementations directly, skipping the
stdio transport hop. Used by run_local_demo.py so the demo is executable without a live MCP
client (spec §45), while tests/test_mcp_live.py separately verifies real stdio server startup
(spec §47). The tool business logic exercised here is identical to what the stdio path runs.
"""

from errors import MCPServerUnavailableError, MCPToolError

_OFFLINE_SERVERS: set[str] = set()


def set_server_offline(server_name: str, offline: bool = True) -> None:
    """Simulate a server being offline/unavailable for demo purposes (spec §44)."""
    if offline:
        _OFFLINE_SERVERS.add(server_name)
    else:
        _OFFLINE_SERVERS.discard(server_name)


def _tool_map() -> dict:
    from mcp_servers.banking_server import server as banking
    from mcp_servers.concierge_ops_server import server as concierge_ops
    from mcp_servers.insurance_server import server as insurance
    from mcp_servers.wealth_server import server as wealth

    return {
        "banking": {
            "get_account_summary": banking.get_account_summary,
            "get_transaction_history": banking.get_transaction_history,
            "get_linked_accounts": banking.get_linked_accounts,
        },
        "insurance": {
            "get_policy_details": insurance.get_policy_details,
            "check_coverage_clause": insurance.check_coverage_clause,
            "get_claim_status": insurance.get_claim_status,
        },
        "wealth": {
            "get_portfolio_summary": wealth.get_portfolio_summary,
            "get_investment_product_details": wealth.get_investment_product_details,
            "check_product_suitability": wealth.check_product_suitability,
        },
        "concierge_ops": {
            "get_customer_kyc_status": concierge_ops.get_customer_kyc_status,
            "run_escalation_check": concierge_ops.run_escalation_check,
            "send_customer_notification": concierge_ops.send_customer_notification,
            "write_audit_log": concierge_ops.write_audit_log,
        },
    }


def call_tool_in_process(server_name: str, tool_name: str, arguments: dict):
    if server_name in _OFFLINE_SERVERS:
        import progress

        def _offline():
            raise MCPServerUnavailableError(
                f"{server_name}_server is offline (simulated)"
            )

        return progress.track(
            server_name, "call_tool", tool_name, _offline, transport="in_process"
        )

    tools = _tool_map().get(server_name)
    if tools is None:
        raise MCPServerUnavailableError(f"Unknown MCP server: {server_name!r}")

    tool = tools.get(tool_name)
    if tool is None:
        raise MCPToolError(f"Unknown tool {tool_name!r} on server {server_name!r}")

    # Same timeline event as the stdio path, tagged transport="in_process" so a demo run in this
    # mode is never mistaken for live MCP traffic.
    import progress

    return progress.track(
        server_name,
        "call_tool",
        tool_name,
        lambda: tool(**arguments),
        transport="in_process",
    )
