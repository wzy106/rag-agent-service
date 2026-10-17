"""Stable chunk metadata: chunk_id, section_path, occurrence.

Pure Python (no torch / no model loading) so it can be imported by unit tests.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from langchain_core.documents import Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_WS_RE = re.compile(r"\s+")


def normalize_content(text: str) -> str:
    """Normalize whitespace so id/occurrence survive formatting-only edits."""
    return _WS_RE.sub(" ", text).strip()


def content_hash(text: str) -> str:
    return hashlib.sha1(normalize_content(text).encode("utf-8")).hexdigest()


def stable_chunk_id(
    source: str,
    section_path: str,
    content: str,
    occurrence: int = 0,
) -> str:
    """Deterministic chunk id.

    `source` is part of the hash so the same content in two documents gets two
    different ids. `occurrence` disambiguates exact duplicates within a source.
    """
    raw = f"{source}\0{section_path}\0{normalize_content(content)}\0{occurrence}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def iter_headings(source_text: str) -> list[tuple[int, int, str]]:
    """Return (offset, level, title) for ATX headings outside fenced code."""
    headings: list[tuple[int, int, str]] = []
    in_fence = False
    offset = 0
    for line in source_text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
        elif not in_fence:
            match = _HEADING_RE.match(line.rstrip("\r\n"))
            if match:
                headings.append((offset, len(match.group(1)), match.group(2).strip()))
        offset += len(line)
    return headings


def _path_at(headings: list[tuple[int, int, str]], offset: int) -> str:
    stack: list[tuple[int, str]] = []
    for heading_offset, level, title in headings:
        if heading_offset > offset:
            break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    return " > ".join(title for _, title in stack)


def split_sections(source_text: str) -> list[tuple[int, str]]:
    """Return (heading_offset, full_section_path) for every heading."""
    headings = iter_headings(source_text)
    out: list[tuple[int, str]] = []
    for offset, level, _ in headings:
        out.append((offset, _path_at(headings, offset)))
    return out


def section_path_for_offset(source_text: str, offset: int) -> str:
    if offset < 0:
        return ""
    return _path_at(iter_headings(source_text), offset)


def _normalized_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace while keeping a map back to original offsets."""
    chars: list[str] = []
    index_map: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            chars.append(" ")
            index_map.append(i)
            prev_space = True
        else:
            chars.append(ch)
            index_map.append(i)
            prev_space = False
    # Trim leading/trailing space consistently with normalize_content().
    start = 0
    while start < len(chars) and chars[start] == " ":
        start += 1
    end = len(chars)
    while end > start and chars[end - 1] == " ":
        end -= 1
    return "".join(chars[start:end]), index_map[start:end]


def _find_offset(
    source_text: str,
    content: str,
    cursor: int,
    normalized_source: str,
    source_map: list[int],
) -> int:
    """Find a chunk's original offset, preferring a cursor-ordered match."""
    needle = normalize_content(content)
    if not needle:
        return cursor
    pos = normalized_source.find(needle[:120], cursor)
    if pos < 0:
        pos = normalized_source.find(needle[:120])
    if pos < 0:
        # Last resort: exact substring search on the raw text.
        raw_pos = source_text.find(content[:120], cursor)
        return raw_pos if raw_pos >= 0 else cursor
    return source_map[pos] if pos < len(source_map) else cursor


def enrich_chunks(chunks: list[Document], docs_by_source: dict[str, str]) -> list[Document]:
    """Copy chunks and add stable chunk_id / section_path / chunk_index.

    Uses `metadata["start_index"]` when available (build path); otherwise finds
    the chunk text in the source document (legacy index path).
    """
    occurrence: dict[tuple[str, str], int] = defaultdict(int)
    cursors: dict[str, int] = defaultdict(int)
    normalized: dict[str, tuple[str, list[int]]] = {}
    headings: dict[str, list[tuple[int, int, str]]] = {}

    out: list[Document] = []
    for global_index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        source = metadata.get("source", "")
        source_text = docs_by_source.get(source, "")

        if source not in normalized:
            normalized[source] = _normalized_with_map(source_text)
            headings[source] = iter_headings(source_text)
        normalized_source, source_map = normalized[source]

        offset = metadata.get("start_index")
        if offset is None:
            offset = _find_offset(
                source_text,
                chunk.page_content,
                cursors[source],
                normalized_source,
                source_map,
            )
        cursors[source] = max(cursors[source], int(offset))

        section_path = _path_at(headings[source], int(offset))
        key = (source, content_hash(chunk.page_content))
        current_occurrence = occurrence[key]
        occurrence[key] += 1

        metadata["chunk_id"] = stable_chunk_id(
            source, section_path, chunk.page_content, current_occurrence
        )
        metadata["section_path"] = section_path
        metadata["chunk_index"] = global_index
        out.append(Document(page_content=chunk.page_content, metadata=metadata))
    return out


def derive_chunk_id(doc: Document) -> str:
    """Return an id for legacy chunks that lack chunk_id metadata."""
    metadata = doc.metadata or {}
    existing = metadata.get("chunk_id")
    if existing:
        return existing
    source = metadata.get("source", "")
    section_path = metadata.get("section_path", "")
    return stable_chunk_id(source, section_path, doc.page_content)
