"""Structured JSON logging + @trace decorator (spec §33)."""

import dataclasses
import functools
import inspect
import json
import logging
import time
import traceback
import uuid
from pathlib import Path

from config import settings
from guardrails import scrub_identifiers

_PII_ARG_NAMES = {
    "customer_id",
    "account_id",
    "policy_id",
    "portfolio_id",
    "claim_id",
    "product_id",
    "phone",
    "email",
    "customer_name",
    "message",
    "customer_stated_intent",
    "customer_message",
    "prior_interaction_summary",
    "raw_text",
    "sanitized_text",
    "text",
    "classified_intent",
    # Drafts can quote the customer's own words back to them (seen in the live-LLM run, TC-095).
    "draft",
    "draft_text",
    "retrieved_context_summary",
}

# Values that are masked completely (no visible tail): raw KYC must never reach logs (spec §24),
# and credentials must never be logged in any form -- not even their last four characters.
_FULLY_MASKED_NAMES = {
    "kyc_status",
    "api_key",
    "anthropic_api_key",
    "ANTHROPIC_API_KEY",
    "authorization",
    "x-api-key",
    "password",
    "secret",
    "token",
}

_MAX_REDACTION_DEPTH = 8


def _mask_value(value) -> str:
    if value is None:
        return "null"
    text = str(value)
    if len(text) <= 4:
        return "****"
    return "****" + text[-4:]


def _redact_value(value, key: str | None = None, depth: int = 0):
    """Recursively redact a value destined for a log line.

    PII-named keys are masked wherever they occur (top-level arguments, nested dicts such as
    MCP `arguments`, Pydantic models, dataclasses, lists). Everything else is returned as-is.
    """
    if key in _FULLY_MASKED_NAMES:
        return "****"
    if key in _PII_ARG_NAMES:
        return _mask_value(value)
    if depth > _MAX_REDACTION_DEPTH:
        return "...<depth limit>"

    if (
        not isinstance(value, type)
        and hasattr(value, "model_dump")
        and callable(value.model_dump)
    ):
        try:
            value = value.model_dump()
        except Exception:  # pragma: no cover - defensive
            return _mask_value(value)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}

    if isinstance(value, dict):
        return {
            str(k): _redact_value(v, str(k), depth + 1)
            for k, v in value.items()
            if k != "self"
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_redact_value(v, None, depth + 1) for v in value]
    return value


def redact_for_logging(args: dict) -> dict:
    """Redact PII-shaped argument names before they hit a log line (spec §32)."""
    return {
        key: _redact_value(value, key) for key, value in args.items() if key != "self"
    }


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "payload", None)
        if payload is None:
            payload = {"event": "LOG", "message": record.getMessage()}
        elif payload.get("event") == "LOG":
            # Ad-hoc LOG events carry structured fields in `payload`; keep the human-readable
            # log message too, otherwise e.g. "LLM drafting failed" is invisible in the sink.
            payload.setdefault("message", record.getMessage())
        payload.setdefault("level", record.levelname)
        payload.setdefault("logger", record.name)
        payload.setdefault("timestamp", self.formatTime(record))
        # Last line of defence: no raw identifier/email/PII and no credential (API key, bearer
        # token) reaches a log sink, whichever code path produced the record (trace args, return
        # previews, tracebacks, SDK exception text, ad-hoc LOG events).
        return scrub_identifiers(json.dumps(payload, default=str))


def _resolve_level(name: str) -> int:
    level = logging.getLevelName(str(name).strip().upper())
    return level if isinstance(level, int) else logging.DEBUG


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(_resolve_level(settings.LOG_LEVEL))

    Path(settings.LOG_DIR).mkdir(parents=True, exist_ok=True)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_JsonFormatter())
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        Path(settings.LOG_DIR) / "nexus.jsonl", encoding="utf-8"
    )
    file_handler.setFormatter(_JsonFormatter())
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


ESCALATION_LOG_FILE = "escalations.jsonl"


def get_escalation_logger() -> logging.Logger:
    """Dedicated, always-on log of escalations: one JSON line per escalated request in
    LOG_DIR/escalations.jsonl (and on stderr).

    It is an operational record, not debug output, so it logs at INFO whatever LOG_LEVEL says
    (LOG_LEVEL=WARNING must not silence it) and does not propagate into nexus.jsonl. Lines go
    through the same JSON formatter, so customer identifiers are masked and secrets scrubbed.
    The case of record (status, triage context) is the escalation_flags row written via the
    concierge_ops MCP server; this log carries the escalation_id that links the two.
    """
    logger = logging.getLogger("nexus.escalations")
    log_dir = Path(settings.LOG_DIR)
    target = (log_dir / ESCALATION_LOG_FILE).resolve()
    current = getattr(logger, "_nexus_target", None)
    if current == target and logger.handlers:
        return logger
    for handler in list(
        logger.handlers
    ):  # LOG_DIR changed (e.g. tests): re-point the sink
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.INFO)
    log_dir.mkdir(parents=True, exist_ok=True)
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(target, encoding="utf-8"),
    ):
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    logger.propagate = False
    logger._nexus_target = target
    return logger


def _preview(value, limit: int = 300):
    try:
        text = json.dumps(_redact_value(value), default=str)
    except Exception:
        text = str(value)
    text = scrub_identifiers(text)
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def trace(logger: logging.Logger, *, redact_return: bool = False):
    """Decorator that emits structured ENTER/EXIT/FAILED trace events.

    redact_return=True replaces the EXIT return preview with a length-only marker; use it for
    functions whose return value embeds customer free text verbatim (e.g. rendered prompts).
    """

    def decorator(func):
        sig = inspect.signature(func)

        def _enter(call_id, args, kwargs):
            try:
                bound = sig.bind_partial(*args, **kwargs).arguments
            except TypeError:
                bound = {"args": list(args), "kwargs": dict(kwargs)}
            logger.debug(
                "ENTER",
                extra={
                    "payload": {
                        "event": "ENTER",
                        "function": func.__qualname__,
                        "call_id": call_id,
                        "args": redact_for_logging(bound),
                    }
                },
            )

        def _failed(call_id, start, exc):
            logger.error(
                "FAILED",
                extra={
                    "payload": {
                        "event": "FAILED",
                        "function": func.__qualname__,
                        "call_id": call_id,
                        "duration_ms": round((time.perf_counter() - start) * 1000, 3),
                        "exception_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    }
                },
            )

        def _exit(call_id, start, result):
            if redact_return:
                preview = f"****<redacted {type(result).__name__}, {len(result) if hasattr(result, '__len__') else '?'} chars>"
            else:
                preview = _preview(result)
            logger.debug(
                "EXIT",
                extra={
                    "payload": {
                        "event": "EXIT",
                        "function": func.__qualname__,
                        "call_id": call_id,
                        "duration_ms": round((time.perf_counter() - start) * 1000, 3),
                        "return_preview": preview,
                    }
                },
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            call_id = uuid.uuid4().hex[:12]
            _enter(call_id, args, kwargs)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                _failed(call_id, start, exc)
                raise
            _exit(call_id, start, result)
            return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            call_id = uuid.uuid4().hex[:12]
            _enter(call_id, args, kwargs)
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
            except Exception as exc:
                _failed(call_id, start, exc)
                raise
            _exit(call_id, start, result)
            return result

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
