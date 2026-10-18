"""Central configuration for Nexus (spec §48). Keep config outside business logic."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
# Per-request timeout and retry budget for Claude calls. Without an explicit timeout the SDK
# default is 10 minutes, which would stall a customer request indefinitely on a slow network.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))
if LLM_TIMEOUT_SECONDS <= 0:
    raise ValueError(f"LLM_TIMEOUT_SECONDS must be positive, got {LLM_TIMEOUT_SECONDS}")
if LLM_MAX_RETRIES < 0:
    raise ValueError(f"LLM_MAX_RETRIES must be >= 0, got {LLM_MAX_RETRIES}")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# auto | sentence-transformers | hashing. "auto" uses Sentence-Transformers when the model can be
# loaded and otherwise the deterministic hashing fallback (see retrieval/embeddings.py).
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto").strip().lower()

CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "data" / "chroma_data"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "nexus_knowledge_base")

DATABASE_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "data" / "meridian.db"))

# Normalised so "debug"/"Debug" work; an unknown name falls back to DEBUG in logging_config.
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").strip().upper() or "DEBUG"
LOG_DIR = os.getenv("LOG_DIR", str(BASE_DIR / "logs"))

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
if not 0.0 <= CONFIDENCE_THRESHOLD <= 1.0:
    raise ValueError(
        f"CONFIDENCE_THRESHOLD must be between 0 and 1, got {CONFIDENCE_THRESHOLD}"
    )
# Confidence ceiling applied by the Escalation Agent when the customer's message carries
# emotional-distress language but matched no hard category (must be below the threshold).
_DISTRESS_CAP_ENV = os.getenv("DISTRESS_CONFIDENCE_CAP", "").strip()
DISTRESS_CONFIDENCE_CAP = (
    float(_DISTRESS_CAP_ENV)
    if _DISTRESS_CAP_ENV
    else min(0.5, CONFIDENCE_THRESHOLD / 2)
)
if not 0.0 <= DISTRESS_CONFIDENCE_CAP <= 1.0 or (
    CONFIDENCE_THRESHOLD > 0 and DISTRESS_CONFIDENCE_CAP >= CONFIDENCE_THRESHOLD
):
    raise ValueError(
        "DISTRESS_CONFIDENCE_CAP must be between 0 and 1 and below CONFIDENCE_THRESHOLD"
    )
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "4000"))
if MAX_INPUT_LENGTH <= 0:
    raise ValueError(f"MAX_INPUT_LENGTH must be positive, got {MAX_INPUT_LENGTH}")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
if RETRIEVAL_TOP_K <= 0:
    raise ValueError(f"RETRIEVAL_TOP_K must be positive, got {RETRIEVAL_TOP_K}")

# Upper bounds for a single MCP call (including a cold server start) and for the initial
# stdio handshake, so a hung server surfaces as MCPServerUnavailableError instead of a stall.
MCP_CALL_TIMEOUT_SECONDS = float(os.getenv("MCP_CALL_TIMEOUT_SECONDS", "90"))
MCP_CONNECT_TIMEOUT_SECONDS = float(os.getenv("MCP_CONNECT_TIMEOUT_SECONDS", "60"))
if MCP_CALL_TIMEOUT_SECONDS <= 0 or MCP_CONNECT_TIMEOUT_SECONDS <= 0:
    raise ValueError(
        "MCP_CALL_TIMEOUT_SECONDS and MCP_CONNECT_TIMEOUT_SECONDS must be positive"
    )
# Extra attempts (fresh process each) when an MCP server's stdio handshake times out on a cold start.
MCP_CONNECT_RETRIES = int(os.getenv("MCP_CONNECT_RETRIES", "1"))
if MCP_CONNECT_RETRIES < 0:
    raise ValueError(f"MCP_CONNECT_RETRIES must be >= 0, got {MCP_CONNECT_RETRIES}")

# Relevance cut-off for knowledge-base hits (BUG-32). Chroma returns squared-L2 distances over
# unit-normalised vectors (0 = identical, 2 = unrelated/orthogonal, 4 = opposite); hits farther than
# the cut-off are dropped instead of always returning top-k. Empty = backend default (see
# retrieval/chroma_store.py: 1.5 for Sentence-Transformers, 1.95 for the lexical hashing fallback,
# which then drops chunks sharing no vocabulary with the query). 0 disables the cut-off.
_MAX_DISTANCE_ENV = os.getenv("RETRIEVAL_MAX_DISTANCE", "").strip()
RETRIEVAL_MAX_DISTANCE = float(_MAX_DISTANCE_ENV) if _MAX_DISTANCE_ENV else None
if RETRIEVAL_MAX_DISTANCE is not None and RETRIEVAL_MAX_DISTANCE < 0:
    raise ValueError(
        f"RETRIEVAL_MAX_DISTANCE must be >= 0, got {RETRIEVAL_MAX_DISTANCE}"
    )
# Upper bound on identifiers of one type looked up per request (a customer may name two accounts;
# an attacker must not be able to fan one request out into dozens of MCP calls).
MAX_IDENTIFIERS_PER_TYPE = int(os.getenv("MAX_IDENTIFIERS_PER_TYPE", "3"))
if MAX_IDENTIFIERS_PER_TYPE <= 0:
    raise ValueError(
        f"MAX_IDENTIFIERS_PER_TYPE must be positive, got {MAX_IDENTIFIERS_PER_TYPE}"
    )

# How process_request runs the five agent roles:
#   crewai_flow -- the request executes as a CrewAI Flow; each step is owned by one CrewAI Agent
#                  role (default; falls back to "direct" only if CrewAI cannot be imported);
#   direct      -- the same stage functions are called in sequence without CrewAI.
NEXUS_ORCHESTRATION = (
    os.getenv("NEXUS_ORCHESTRATION", "crewai_flow").strip().lower() or "crewai_flow"
)
if NEXUS_ORCHESTRATION not in ("crewai_flow", "direct"):
    raise ValueError(
        f"NEXUS_ORCHESTRATION must be 'crewai_flow' or 'direct', got {NEXUS_ORCHESTRATION!r}"
    )
# When the LLM is configured, fetch the classification/drafting prompt templates from the
# concierge_ops MCP server (falls back to the inline templates if the server is unreachable).
NEXUS_MCP_PROMPTS = os.getenv("NEXUS_MCP_PROMPTS", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Send the customer a handoff notification (KYC-gated, audited, via MCP) on escalation.
NOTIFY_ON_ESCALATION = os.getenv("NOTIFY_ON_ESCALATION", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Sliding-window rate limits for process_request (0 disables): per customer id and overall.
# They bound enumeration of customer ids and runaway LLM/MCP cost (scorecard 4.6, S11).
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
RATE_LIMIT_GLOBAL_PER_MINUTE = int(os.getenv("RATE_LIMIT_GLOBAL_PER_MINUTE", "120"))
if RATE_LIMIT_PER_MINUTE < 0 or RATE_LIMIT_GLOBAL_PER_MINUTE < 0:
    raise ValueError(
        "RATE_LIMIT_PER_MINUTE and RATE_LIMIT_GLOBAL_PER_MINUTE must be >= 0"
    )
# Streamlit: distinct customer ids one browser session may query before it must be restarted.
UI_MAX_CUSTOMERS_PER_SESSION = int(os.getenv("UI_MAX_CUSTOMERS_PER_SESSION", "3"))
# Start the four MCP server processes in parallel, in the background, when the UI starts, so the
# first customer request does not pay ~4 sequential cold starts (scorecard 4.1 latency).
MCP_PREWARM = os.getenv("MCP_PREWARM", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Server-side switch for the "full" field scope (balances/holdings) on MCP tools. Off by default:
# the concierge flow only ever needs the minimal retrieval scope, and a client-asserted
# caller_scope must not be able to widen visibility on its own.
MCP_FULL_SCOPE_ENABLED = os.getenv("MCP_FULL_SCOPE_ENABLED", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"

HARD_ESCALATION_CATEGORIES = {
    "financial_hardship",
    "safeguarding_concern",
    "compliance_override_request",
}

# Non-customer review categories that ALWAYS route to a human, checked after the hard customer
# categories (which keep precedence so e.g. hardship + injection reaches the hardship team) and
# before confidence. A message carrying a prompt-injection signature is quarantined: it never
# reaches the LLM, retrieval or drafting, and is handed to a specialist for security review.
SECURITY_REVIEW_CATEGORIES = {
    "prompt_injection_suspected",
}
