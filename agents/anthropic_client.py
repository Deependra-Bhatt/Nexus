"""Thin wrapper around the Anthropic Claude client (spec §49).

Request shape follows the anthropic 1.x SDK / current Claude models:
- no sampling parameters (`temperature`/`top_p`/`top_k` were removed from the SDK and are
  rejected by Claude Sonnet 5.5 and later; sending them is what silently disabled every LLM
  call in this project before the audit);
- reasoning depth is controlled with `output_config.effort`;
- JSON-returning agents use structured outputs (`output_config.format` = json_schema) so the
  model cannot return malformed or extra fields;
- a `refusal` or `max_tokens` stop reason is surfaced as an error so callers fall back to
  their deterministic path instead of using partial/declined content.
"""

import json
from functools import lru_cache

from config import settings


@lru_cache(maxsize=1)
def get_client():
    import anthropic

    # Explicit timeout/retry budget: the SDK default timeout is 10 minutes, which would stall a
    # customer request indefinitely on a slow or blocked network.
    return anthropic.Anthropic(
        api_key=settings.ANTHROPIC_API_KEY,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.LLM_MAX_RETRIES,
    )


def is_configured() -> bool:
    return bool(settings.ANTHROPIC_API_KEY)


class ModelResponseError(RuntimeError):
    """The model did not produce usable content (refusal, truncation, empty, malformed)."""


def _request_kwargs(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    effort: str | None,
) -> dict:
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if effort:
        kwargs["output_config"] = {"effort": effort}
    return kwargs


def _text_of(response) -> str:
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details is not None else None
        raise ModelResponseError(
            f"Model declined the request (refusal, category={category!r})"
        )
    if stop_reason == "max_tokens":
        raise ModelResponseError("Model output was truncated at max_tokens")
    return "".join(
        block.text
        for block in getattr(response, "content", [])
        if getattr(block, "type", None) == "text"
    )


def complete_json(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    effort: str | None = "low",
    schema: dict | None = None,
) -> dict:
    """Call Claude and return a JSON object. With `schema`, structured outputs constrain the
    response to that JSON Schema; without it, the first {...} object in the text is parsed.
    """
    client = get_client()
    kwargs = _request_kwargs(system_prompt, user_prompt, model, max_tokens, effort)
    if schema is not None:
        kwargs.setdefault("output_config", {})["format"] = {
            "type": "json_schema",
            "schema": schema,
        }
    response = client.messages.create(**kwargs)
    data = _extract_json(_text_of(response))
    if not isinstance(data, dict):
        raise ModelResponseError("Model output is not a JSON object")
    return data


def complete_text(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    effort: str | None = "medium",
) -> str:
    client = get_client()
    response = client.messages.create(
        **_request_kwargs(system_prompt, user_prompt, model, max_tokens, effort)
    )
    text = _text_of(response).strip()
    if not text:
        raise ModelResponseError("Model returned an empty response")
    return text


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ModelResponseError(
            f"No JSON object found in model output: {text[:200]!r}"
        )
    return json.loads(text[start : end + 1])
