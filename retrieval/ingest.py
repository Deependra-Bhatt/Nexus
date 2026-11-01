"""Document loading, chunking and metadata assignment (spec §13)."""

import re
from dataclasses import dataclass, field
from pathlib import Path

from config import settings

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
_CHUNK_SIZE = 600
_CHUNK_OVERLAP = 80


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    document_type: str
    source_path: str
    product_line: str
    section: str
    version: str
    text: str
    metadata: dict = field(default_factory=dict)


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    front, body = match.groups()
    meta = {}
    for line in front.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, body.strip()


def _chunk_text(
    text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP
) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def load_documents(base_dir: Path | None = None) -> list[DocumentChunk]:
    base_dir = base_dir or settings.KNOWLEDGE_BASE_DIR
    chunks: list[DocumentChunk] = []
    for path in sorted(Path(base_dir).rglob("*.md")):
        # utf-8-sig drops a BOM and CRLF is normalised so frontmatter parses the same on every
        # platform/checkout (git autocrlf would otherwise silently strip all document metadata).
        raw = (
            path.read_text(encoding="utf-8-sig")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )
        meta, body = _parse_frontmatter(raw)
        document_id = meta.get("document_id", path.stem)
        document_type = meta.get("document_type", "unknown")
        product_line = meta.get("product_line", "unknown")
        section = meta.get("section", "unknown")
        version = meta.get("version", "1.0")

        for idx, piece in enumerate(_chunk_text(body)):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document_id}-{idx}",
                    document_id=document_id,
                    document_type=document_type,
                    source_path=str(path),
                    product_line=product_line,
                    section=section,
                    version=version,
                    text=piece,
                    metadata={
                        "document_id": document_id,
                        "document_type": document_type,
                        "source_path": str(path),
                        "product_line": product_line,
                        "section": section,
                        "version": version,
                    },
                )
            )
    return chunks
