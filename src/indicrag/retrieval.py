"""Lexical, dense, and hybrid passage retrieval."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import Passage, SearchHit

RetrievalMethod = Literal["bm25", "tfidf", "dense", "hybrid"]
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/LaBSE"
TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return TOKEN_RE.findall(normalized)


class HybridRetriever:
    """BM25 + character TF-IDF + LaBSE retrieval with transparent scores.

    The default hybrid score is a weighted blend of normalized lexical and
    multilingual dense scores. The alpha parameter is the lexical weight.
    """

    def __init__(
        self,
        passages: list[Passage],
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        alpha: float = 0.55,
    ) -> None:
        if not passages:
            raise ValueError("At least one passage is required.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        self.passages = passages
        self.embedding_model_name = embedding_model
        self.alpha = alpha
        self._doc_tokens = [_tokens(item.text) for item in passages]
        self._avg_doc_len = max(1.0, sum(map(len, self._doc_tokens)) / len(self._doc_tokens))
        self._document_frequency = Counter(
            token for document in self._doc_tokens for token in set(document)
        )
        self._tfidf = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True,
            lowercase=True,
            dtype=np.float32,
        )
        self._tfidf_matrix = self._tfidf.fit_transform([item.text for item in passages])
        self._dense_model = None
        self._dense_matrix: np.ndarray | None = None

    def _bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        query_terms = _tokens(query)
        scores = np.zeros(len(self.passages), dtype=np.float32)
        if not query_terms:
            return scores
        n_docs = len(self.passages)
        for index, document in enumerate(self._doc_tokens):
            frequencies = Counter(document)
            length_norm = k1 * (1.0 - b + b * len(document) / self._avg_doc_len)
            score = 0.0
            for term in query_terms:
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                df = self._document_frequency.get(term, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                score += idf * (tf * (k1 + 1.0)) / (tf + length_norm)
            scores[index] = score
        return scores

    def _dense_embeddings(self) -> np.ndarray:
        if self._dense_matrix is not None:
            return self._dense_matrix
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval requires sentence-transformers. Install requirements.txt."
            ) from exc
        self._dense_model = SentenceTransformer(self.embedding_model_name)
        self._dense_matrix = self._dense_model.encode(
            [item.text for item in self.passages],
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return self._dense_matrix

    def search(
        self,
        query: str,
        *,
        method: RetrievalMethod = "hybrid",
        top_k: int = 5,
    ) -> list[SearchHit]:
        if not query.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if method not in {"bm25", "tfidf", "dense", "hybrid"}:
            raise ValueError(f"Unknown retrieval method: {method}")

        n = len(self.passages)
        bm25_raw = self._bm25(query)
        bm25_conf = 1.0 - np.exp(-bm25_raw / 4.0)
        query_vector = self._tfidf.transform([query])
        tfidf_scores = cosine_similarity(query_vector, self._tfidf_matrix).ravel()
        lexical_scores = np.maximum(bm25_conf, tfidf_scores)
        components: dict[str, np.ndarray] = {
            "bm25": bm25_conf,
            "tfidf": tfidf_scores,
            "lexical": lexical_scores,
        }

        if method in {"dense", "hybrid"}:
            matrix = self._dense_embeddings()
            query_embedding = self._dense_model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            dense_scores = np.clip(matrix @ query_embedding, 0.0, 1.0)
            components["dense"] = dense_scores

        if method == "bm25":
            scores = bm25_conf
        elif method == "tfidf":
            scores = tfidf_scores
        elif method == "dense":
            scores = components["dense"]
        else:
            scores = self.alpha * lexical_scores + (1.0 - self.alpha) * components["dense"]

        order = np.argsort(-scores, kind="stable")[: min(top_k, n)]
        hits: list[SearchHit] = []
        for rank, index in enumerate(order, start=1):
            values = {name: float(value[index]) for name, value in components.items()}
            hits.append(
                SearchHit(
                    passage=self.passages[int(index)],
                    rank=rank,
                    score=float(scores[index]),
                    component_scores=values,
                )
            )
        return hits
