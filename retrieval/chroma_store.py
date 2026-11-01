"""Chroma vector store wrapper (spec §13)."""

import threading
from functools import lru_cache

from config import settings
from logging_config import get_logger
from retrieval.embeddings import embed_texts, embedding_backend
from retrieval.ingest import load_documents

logger = get_logger("nexus.chroma_store")

_BACKEND_KEY = "embedding_backend"
# Default relevance cut-off per embedding backend (squared-L2 over unit vectors; see settings).
_DEFAULT_MAX_DISTANCE = {"sentence-transformers": 1.5, "hashing-fallback": 1.95}
# Serialises check-then-rebuild so concurrent searches never observe a deleted/half-built
# collection (Streamlit serves requests from several threads).
_INDEX_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def get_client():
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    # anonymized_telemetry=False: no outbound telemetry from a financial-services component.
    return chromadb.PersistentClient(
        path=settings.CHROMA_PATH, settings=ChromaSettings(anonymized_telemetry=False)
    )


def get_collection():
    client = get_client()
    return client.get_or_create_collection(name=settings.CHROMA_COLLECTION)


def _collection_backend(collection) -> str | None:
    return (collection.metadata or {}).get(_BACKEND_KEY)


def build_index(force: bool = False) -> int:
    """Ingest knowledge_base documents into Chroma. Returns number of chunks indexed.

    The collection records which embedding backend produced its vectors. If the active backend
    differs (e.g. the index was built with the hashing fallback and Sentence-Transformers is now
    available), the index is rebuilt rather than silently queried with incompatible vectors.
    """
    with _INDEX_LOCK:
        client = get_client()
        collection = get_collection()
        backend = embedding_backend()
        recorded = _collection_backend(collection)

        if collection.count() > 0 and not force and recorded == backend:
            return collection.count()

        if collection.count() > 0 and recorded != backend and not force:
            logger.error(
                "Knowledge-base index was built with a different embedding backend; rebuilding",
                extra={
                    "payload": {
                        "event": "LOG",
                        "recorded_backend": recorded,
                        "active_backend": backend,
                    }
                },
            )

        client.delete_collection(settings.CHROMA_COLLECTION)
        collection = client.create_collection(
            name=settings.CHROMA_COLLECTION, metadata={_BACKEND_KEY: backend}
        )

        chunks = load_documents()
        if not chunks:
            return 0

        embeddings = embed_texts([c.text for c in chunks])
        collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        return len(chunks)


def _ready_collection():
    """Return the collection, building/rebuilding it first if it is empty or backend-mismatched."""
    with _INDEX_LOCK:
        collection = get_collection()
        if (
            collection.count() == 0
            or _collection_backend(collection) != embedding_backend()
        ):
            build_index()
            collection = get_collection()
        return collection


def semantic_search(
    query: str, top_k: int | None = None, product_line: str | None = None
) -> list[dict]:
    top_k = top_k if isinstance(top_k, int) and top_k > 0 else settings.RETRIEVAL_TOP_K
    with _INDEX_LOCK:
        collection = _ready_collection()
        total = collection.count()
        if total == 0:
            return []
        where = None
        if product_line and product_line != "cross_product":
            where = {"product_line": {"$in": [product_line, "cross_product"]}}

        result = collection.query(
            query_embeddings=embed_texts([query]),
            n_results=min(top_k, total),
            where=where,
        )

    hits = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    max_distance = settings.RETRIEVAL_MAX_DISTANCE
    if max_distance is None:
        max_distance = _DEFAULT_MAX_DISTANCE.get(embedding_backend(), 1.95)
    dropped = 0
    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        # Relevance cut-off (BUG-32): an unrelated chunk is no grounding at all, so it is dropped
        # instead of being passed to drafting just because top-k always returns something.
        if max_distance and distance is not None and distance > max_distance:
            dropped += 1
            continue
        hits.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": meta or {},
                "distance": distance,
            }
        )
    if dropped:
        logger.debug(
            "Knowledge-base hits below the relevance cut-off were dropped",
            extra={
                "payload": {
                    "event": "LOG",
                    "dropped": dropped,
                    "kept": len(hits),
                    "max_distance": max_distance,
                }
            },
        )
    return hits


def get_document_excerpt(document_id: str) -> dict | None:
    """Return a single sanitized excerpt for a document_id (used by the knowledgebase resource)."""
    # A lookup by document_id compares no vectors, so it does not need (or load) the embedding
    # model. Calling _ready_collection() here made the concierge_ops server import
    # Sentence-Transformers/torch on its first knowledgebase:// read -- over a minute on a cold
    # Windows start, which exceeded MCP_CALL_TIMEOUT_SECONDS (TC-153). Build only if empty.
    with _INDEX_LOCK:
        collection = get_collection()
        if collection.count() == 0:
            build_index()
            collection = get_collection()
        result = collection.get(where={"document_id": document_id}, limit=1)
    ids = result.get("ids", [])
    if not ids:
        return None
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    text = documents[0] if documents else ""
    meta = (metadatas[0] if metadatas else {}) or {}

    import guardrails

    excerpt = guardrails.sanitize_retrieved_text(text)[:500]
    return {
        "document_id": document_id,
        "document_type": meta.get("document_type"),
        "section": meta.get("section"),
        "excerpt": excerpt,
    }
