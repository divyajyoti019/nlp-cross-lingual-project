"""Document extraction and stable passage segmentation."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

from .schema import Passage

DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
HEADING_RE = re.compile(r"^\s*(#{1,6}\s+.+|(?:\d+(?:\.\d+)*\.?)\s+.{2,140})\s*$")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    """Return a lightweight script label, not a linguistic language ID."""
    devanagari = len(DEVANAGARI_RE.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if devanagari and latin:
        return "code_mixed"
    if devanagari:
        return "hi"
    return "en"


def _split_blocks(text: str) -> list[tuple[str, str]]:
    current_heading = ""
    blocks: list[tuple[str, str]] = []
    paragraph_lines: list[str] = []

    def flush() -> None:
        nonlocal paragraph_lines
        value = clean_text(" ".join(line.strip() for line in paragraph_lines if line.strip()))
        if value:
            blocks.append((current_heading, value))
        paragraph_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
        elif HEADING_RE.match(stripped):
            flush()
            current_heading = stripped.lstrip("#").strip()
        else:
            paragraph_lines.append(stripped)
    flush()
    return blocks


def _chunks(text: str, max_chars: int, overlap_chars: int) -> Iterable[str]:
    text = clean_text(text)
    if len(text) <= max_chars:
        if text:
            yield text
        return

    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind(" ", start, end), text.rfind("।", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)


def _read_file(path: Path) -> list[tuple[int | None, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF ingestion requires pypdf. Install requirements.txt.") from exc
        reader = PdfReader(str(path))
        return [(index + 1, page.extract_text() or "") for index, page in enumerate(reader.pages)]
    if suffix in {".txt", ".md"}:
        return [(None, path.read_text(encoding="utf-8", errors="replace"))]
    raise ValueError(f"Unsupported file type {suffix!r}; use PDF, TXT, or Markdown.")


def ingest_file(
    path: str | Path,
    *,
    doc_id: str | None = None,
    title: str | None = None,
    source_url: str = "",
    language: str = "auto",
    max_chars: int = 900,
    overlap_chars: int = 120,
) -> list[Passage]:
    """Extract a PDF/TXT/Markdown file into passages with source metadata."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if max_chars < 200 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("Use max_chars >= 200 and 0 <= overlap_chars < max_chars.")

    document_id = doc_id or re.sub(r"[^a-zA-Z0-9_-]+", "-", source.stem).strip("-").lower()
    document_title = title or source.stem.replace("_", " ").replace("-", " ").strip()
    passages: list[Passage] = []
    chunk_number = 0

    for page, raw_text in _read_file(source):
        for section, block in _split_blocks(raw_text):
            for chunk in _chunks(block, max_chars, overlap_chars):
                chunk_number += 1
                digest = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:8]
                page_part = f"-p{page:03d}" if page is not None else ""
                section_part = re.sub(r"[^a-zA-Z0-9]+", "-", section).strip("-").lower()[:24]
                section_part = f"-{section_part}" if section_part else ""
                passage_id = f"{document_id}{page_part}{section_part}-c{chunk_number:04d}-{digest}"
                passages.append(
                    Passage(
                        passage_id=passage_id,
                        doc_id=document_id,
                        title=document_title,
                        source_url=source_url,
                        language=detect_language(chunk) if language == "auto" else language,
                        section=section,
                        text=chunk,
                        page=page,
                    )
                )
    return passages


def load_passages(path: str | Path) -> list[Passage]:
    source = Path(path)
    rows = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(Passage.from_dict(json.loads(line)))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid passage JSONL at {source}:{line_number}: {exc}") from exc
    return rows


def write_passages(passages: Iterable[Passage], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for passage in passages:
            stream.write(json.dumps(passage.to_dict(), ensure_ascii=False) + "\n")
    return destination
