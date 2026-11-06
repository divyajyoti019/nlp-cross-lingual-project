"""Typed data structures shared by ingestion, retrieval, and answer generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Passage:
    passage_id: str
    doc_id: str
    title: str
    source_url: str
    language: str
    section: str
    text: str
    page: int | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Passage":
        return cls(
            passage_id=str(row["passage_id"]),
            doc_id=str(row.get("doc_id", "")),
            title=str(row.get("title", row.get("doc_id", "Untitled document"))),
            source_url=str(row.get("source_url", "")),
            language=str(row.get("language", "unknown")),
            section=str(row.get("section", "")),
            text=str(row["text"]),
            page=int(row["page"]) if row.get("page") not in (None, "") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchHit:
    passage: Passage
    rank: int
    score: float
    component_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passage": self.passage.to_dict(),
            "rank": self.rank,
            "score": self.score,
            "component_scores": self.component_scores,
        }
