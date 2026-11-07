"""Prompt templates served by the concierge_ops MCP server (spec §26).

When the LLM is configured, the Classifier and Drafting agents render their user prompt from the
MCP prompts `request_classification_prompt` / `response_drafting_prompt`, so the templates are
owned and versioned by the MCP layer. If the server cannot render a prompt, the agent's inline
template is used and the fallback is recorded as degraded behaviour (never silent).
Controlled by NEXUS_MCP_PROMPTS (on by default; the deterministic test suite pins it off).
"""

from agents import degradation
from config import settings
from logging_config import get_logger
from mcp_servers.client import get_prompt_sync

logger = get_logger("nexus.agents.prompts")


def render(prompt_name: str, arguments: dict, fallback: str) -> str:
    if not settings.NEXUS_MCP_PROMPTS:
        return fallback
    try:
        text = get_prompt_sync(
            "concierge_ops", prompt_name, {k: str(v) for k, v in arguments.items()}
        )
    except Exception as exc:
        logger.warning(
            "MCP prompt unavailable; using inline template",
            extra={
                "payload": {
                    "event": "LOG",
                    "degraded": "prompts",
                    "prompt": prompt_name,
                    "error": str(exc),
                }
            },
        )
        degradation.record("prompts", f"{prompt_name}_inline_fallback")
        return fallback
    return text or fallback
