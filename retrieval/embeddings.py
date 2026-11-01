"""Sentence-Transformer embedding wrapper (spec §13).

Sentence-Transformers is the configured embedding backend. When its model weights cannot be
loaded -- e.g. the HuggingFace download is blocked by a corporate proxy, or the host is
offline -- this module falls back to a deterministic local hashing embedding so the retrieval
pipeline, demo and tests remain runnable. The active backend is reported by
`embedding_backend()` and logged, so a degraded run is never mistaken for a full one.

Importing sentence-transformers (and torch) costs well over a minute on this machine, so when
the outcome is already known -- HF_HUB_OFFLINE is set and the model is not in the local cache,
or EMBEDDING_BACKEND=hashing -- the import is skipped entirely.
"""

import hashlib
import math
import os
import re
from functools import lru_cache
from pathlib import Path

from config import settings
from logging_config import get_logger

logger = get_logger("nexus.embeddings")

EMBEDDING_DIM = (
    384  # matches all-MiniLM-L6-v2 so a Chroma collection stays dimension-stable
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TRUTHY = ("1", "true", "yes", "on")


def _enable_os_trust_store() -> None:
    """Use the OS certificate store so corporate TLS interception does not break downloads."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass


def _forced_backend() -> str | None:
    value = (settings.EMBEDDING_BACKEND or "auto").strip().lower()
    if value in ("hashing", "hash", "fallback", "hashing-fallback"):
        return "hashing-fallback"
    if value in ("sentence-transformers", "sentence_transformers", "st"):
        return "sentence-transformers"
    return None


def _hf_cache_dir() -> Path:
    if os.getenv("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.getenv("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _offline_without_cached_model() -> bool:
    """True when HF_HUB_OFFLINE is set and the configured model has no local snapshot, i.e. the
    Sentence-Transformers load is guaranteed to fail and the import can be skipped."""
    if os.getenv("HF_HUB_OFFLINE", "").strip().lower() not in _TRUTHY:
        return False
    name = settings.EMBEDDING_MODEL
    if Path(name).is_dir():
        return False
    repo = name if "/" in name else f"sentence-transformers/{name}"
    snapshots = _hf_cache_dir() / f"models--{repo.replace('/', '--')}" / "snapshots"
    try:
        return not (snapshots.is_dir() and any(snapshots.iterdir()))
    except OSError:
        return True


def _log_fallback(reason: str) -> None:
    logger.error(
        "Sentence-Transformers model unavailable; using deterministic local fallback "
        "embeddings. Semantic quality is reduced.",
        extra={
            "payload": {
                "event": "LOG",
                "backend": "hashing-fallback",
                "model": settings.EMBEDDING_MODEL,
                "reason": reason,
            }
        },
    )


@lru_cache(maxsize=1)
def get_embedding_model():
    """Return a loaded SentenceTransformer, or None if the model is unavailable."""
    forced = _forced_backend()
    if forced == "hashing-fallback":
        logger.info(
            "Embedding backend forced to hashing fallback by EMBEDDING_BACKEND",
            extra={"payload": {"event": "LOG", "backend": "hashing-fallback"}},
        )
        return None
    if forced is None and _offline_without_cached_model():
        _log_fallback(
            "HF_HUB_OFFLINE is set and the model is not in the local HuggingFace cache"
        )
        return None

    _enable_os_trust_store()
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.debug(
            "Embedding backend ready",
            extra={
                "payload": {
                    "event": "LOG",
                    "backend": "sentence-transformers",
                    "model": settings.EMBEDDING_MODEL,
                }
            },
        )
        return model
    except Exception as exc:
        _log_fallback(f"{type(exc).__name__}: {exc}")
        return None


def embedding_backend() -> str:
    return (
        "sentence-transformers"
        if get_embedding_model() is not None
        else "hashing-fallback"
    )


def _fallback_embed_text(text: str) -> list[float]:
    """Deterministic hashed bag-of-words vector, L2-normalized for cosine similarity."""
    counts: dict[int, float] = {}
    for token in _TOKEN_RE.findall((text or "").lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        index = int.from_bytes(digest, "big") % EMBEDDING_DIM
        counts[index] = counts.get(index, 0.0) + 1.0

    vector = [0.0] * EMBEDDING_DIM
    for index, count in counts.items():
        vector[index] = 1.0 + math.log(count)  # sublinear term frequency

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedding_model()
    if model is None:
        return [_fallback_embed_text(text) for text in texts]
    # Unit-normalised so squared-L2 distances share one scale across backends (RETRIEVAL_MAX_DISTANCE).
    return model.encode(
        list(texts), convert_to_numpy=True, normalize_embeddings=True
    ).tolist()


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
