"""Per-request record of degraded behaviour (scorecard 4.2: "silent-ish degradation").

When an LLM-backed agent falls back to its deterministic path, it calls `record(...)`. The
orchestrator opens a fresh record for every request and copies it onto
NexusResponse.degraded, so the trace, the UI and the audit log all show that, e.g., the
classification was rule-based -- not only an ops-log WARNING.

A ContextVar holding a *mutable list* is used so the record is shared with work that runs in a
copied context (asyncio tasks / CrewAI Flow steps) for the same request.
"""

from contextvars import ContextVar

_CURRENT: ContextVar[list | None] = ContextVar("nexus_degradation", default=None)


def start() -> list:
    events: list[str] = []
    _CURRENT.set(events)
    return events


def record(component: str, reason: str) -> None:
    events = _CURRENT.get()
    if events is None:
        return
    entry = f"{component}:{reason}"
    if entry not in events:
        events.append(entry)


def current() -> list[str]:
    return list(_CURRENT.get() or [])
