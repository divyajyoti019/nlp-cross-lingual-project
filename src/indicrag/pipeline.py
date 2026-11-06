"""Cross-lingual retrieval, answerability gating, and evidence citations."""

from __future__ import annotations

import re
from typing import Any

from .generation import OllamaChatGenerator, UNANSWERABLE_TEXT, extractive_answer
from .retrieval import DEFAULT_EMBEDDING_MODEL, HybridRetriever, RetrievalMethod
from .schema import Passage

HINGLISH_MARKERS = {
    "hai", "hain", "kya", "ka", "ki", "ke", "ko", "mein", "me", "se", "liye",
    "kaise", "kab", "kitna", "kitne", "kyun", "batao", "bataiye", "hota", "hote",
}
DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")


def detect_query_language(query: str) -> str:
    """Script/lexicon heuristic for reporting language slices."""
    has_devanagari = bool(DEVANAGARI_RE.search(query))
    has_latin = bool(re.search(r"[A-Za-z]", query))
    if has_devanagari and has_latin:
        return "code_mixed"
    if has_devanagari:
        return "hi"
    latin_tokens = {token.casefold() for token in re.findall(r"[A-Za-z]+", query)}
    if len(latin_tokens & HINGLISH_MARKERS) >= 2:
        return "code_mixed"
    return "en"


class IndicRAG:
    def __init__(
        self,
        passages: list[Passage],
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        alpha: float = 0.55,
        ollama: OllamaChatGenerator | None = None,
    ) -> None:
        self.retriever = HybridRetriever(
            passages, embedding_model=embedding_model, alpha=alpha
        )
        self.ollama = ollama

    def retrieve(
        self,
        query: str,
        *,
        method: RetrievalMethod = "hybrid",
        top_k: int = 5,
    ):
        return self.retriever.search(query, method=method, top_k=top_k)

    def ask(
        self,
        query: str,
        *,
        method: RetrievalMethod = "hybrid",
        top_k: int = 5,
        threshold: float = 0.42,
        use_ollama: bool = False,
    ) -> dict[str, Any]:
        hits = self.retrieve(query, method=method, top_k=top_k)
        confidence = hits[0].score if hits else 0.0
        answerable = bool(hits) and confidence >= threshold
        if not answerable:
            answer = UNANSWERABLE_TEXT
        elif use_ollama:
            if self.ollama is None:
                raise RuntimeError("Ollama generation is not configured.")
            answer = self.ollama.answer(query, hits)
            if answer == UNANSWERABLE_TEXT:
                answerable = False
        else:
            answer = extractive_answer(query, hits)

        citations = []
        for hit in hits:
            passage = hit.passage
            citation_url = passage.source_url
            if citation_url and passage.page is not None and citation_url.lower().endswith(".pdf"):
                citation_url = f"{citation_url}#page={passage.page}"
            citations.append(
                {
                    "passage_id": passage.passage_id,
                    "doc_id": passage.doc_id,
                    "title": passage.title,
                    "source_url": citation_url,
                    "section": passage.section,
                    "page": passage.page,
                    "language": passage.language,
                    "excerpt": passage.text,
                    "score": hit.score,
                    "component_scores": hit.component_scores,
                }
            )
        return {
            "query": query,
            "query_language": detect_query_language(query),
            "retrieval_method": method,
            "answerable": answerable,
            "confidence": confidence,
            "threshold": threshold,
            "answer": answer,
            "citations": citations,
        }
