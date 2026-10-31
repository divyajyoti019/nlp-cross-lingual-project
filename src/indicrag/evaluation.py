"""Separate retrieval, answer-quality, and answerability evaluation."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from .generation import OllamaChatGenerator, UNANSWERABLE_TEXT, extractive_answer
from .pipeline import detect_query_language
from .retrieval import HybridRetriever, RetrievalMethod
from .schema import Passage


def load_qa(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid QA JSONL at {path}:{line_number}: {exc}") from exc
            required = {"id", "query", "answerable", "split"}
            missing = required - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} is missing fields: {sorted(missing)}")
            rows.append(row)
    return rows


def normalize_answer(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[\w]+", normalized, flags=re.UNICODE))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common: dict[str, int] = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in ref_tokens:
        if common.get(token, 0):
            overlap += 1
            common[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _binary_metrics(gold: list[bool], predicted: list[bool]) -> dict[str, float]:
    tp = sum(a and b for a, b in zip(gold, predicted))
    fp = sum((not a) and b for a, b in zip(gold, predicted))
    fn = sum(a and (not b) for a, b in zip(gold, predicted))
    tn = sum((not a) and (not b) for a, b in zip(gold, predicted))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, len(gold))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def calibrate_threshold(
    retriever: HybridRetriever,
    rows: list[dict[str, Any]],
    *,
    method: RetrievalMethod,
    top_k: int,
) -> float:
    """Select a relevance cutoff by F1 on a development split only."""
    scored = []
    for row in rows:
        hits = retriever.search(row["query"], method=method, top_k=top_k)
        scored.append((hits[0].score if hits else 0.0, bool(row["answerable"])))
    if not scored:
        raise ValueError("No development rows available for threshold calibration.")
    candidates = [i / 100 for i in range(1, 100)]
    best_threshold = 0.42
    best_key = (-1.0, -1.0)
    for threshold in candidates:
        metrics = _binary_metrics(
            [label for _, label in scored],
            [score >= threshold for score, _ in scored],
        )
        key = (metrics["f1"], threshold)
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold


def _retrieval_metrics(
    rows: list[dict[str, Any]],
    hit_rows: list[list[Any]],
    top_k: int,
) -> dict[str, float]:
    recall_values = []
    precision_values = []
    reciprocal_ranks = []
    evidence_rows = 0
    for row, hits in zip(rows, hit_rows):
        gold_ids = set(row.get("gold_passage_ids", []))
        if not gold_ids:
            continue
        evidence_rows += 1
        ranked_ids = [hit.passage.passage_id for hit in hits[:top_k]]
        relevant_ranks = [index + 1 for index, item in enumerate(ranked_ids) if item in gold_ids]
        relevant_count = len(relevant_ranks)
        recall_values.append(relevant_count / len(gold_ids))
        precision_values.append(relevant_count / top_k)
        reciprocal_ranks.append(1.0 / min(relevant_ranks) if relevant_ranks else 0.0)
    denominator = max(1, evidence_rows)
    return {
        f"recall@{top_k}": sum(recall_values) / denominator,
        f"precision@{top_k}": sum(precision_values) / denominator,
        f"mrr@{top_k}": sum(reciprocal_ranks) / denominator,
        "retrieval_examples": float(evidence_rows),
    }


def _score_rows(
    retriever: HybridRetriever,
    rows: list[dict[str, Any]],
    *,
    method: RetrievalMethod,
    top_k: int,
    threshold: float,
    ollama: OllamaChatGenerator | None,
    compare_direct: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predicted_answerability: list[bool] = []
    gold_answerability: list[bool] = []
    em_values: list[float] = []
    f1_values: list[float] = []
    hit_rows = []
    detail_rows = []

    for row in rows:
        hits = retriever.search(row["query"], method=method, top_k=top_k)
        hit_rows.append(hits)
        score = hits[0].score if hits else 0.0
        gold_answerable = bool(row["answerable"])
        predicted = bool(hits) and score >= threshold
        answer = UNANSWERABLE_TEXT
        if predicted:
            if ollama is not None:
                answer = ollama.answer(row["query"], hits)
                if answer == UNANSWERABLE_TEXT:
                    predicted = False
            else:
                answer = extractive_answer(row["query"], hits)

        predicted_answerability.append(predicted)
        gold_answerability.append(gold_answerable)
        if gold_answerable:
            gold_answer = str(row.get("gold_answer", ""))
            normalized_prediction = normalize_answer(answer)
            normalized_gold = normalize_answer(gold_answer)
            em_values.append(float(normalized_prediction == normalized_gold))
            f1_values.append(token_f1(answer, gold_answer))

        top_hit = hits[0] if hits else None
        detail = {
            "id": row["id"],
            "split": row["split"],
            "query": row["query"],
            "query_language": row.get("query_language") or detect_query_language(row["query"]),
            "question_type": row.get("question_type", ""),
            "hard_case": bool(row.get("hard_case", False)),
            "gold_answerable": gold_answerable,
            "predicted_answerable": predicted,
            "confidence": score,
            "threshold": threshold,
            "answer": answer,
            "gold_answer": row.get("gold_answer", ""),
            "top_passage_id": top_hit.passage.passage_id if top_hit else "",
            "top_source_url": top_hit.passage.source_url if top_hit else "",
            "top_section": top_hit.passage.section if top_hit else "",
            "retrieved_passage_ids": [hit.passage.passage_id for hit in hits],
        }
        if compare_direct:
            if ollama is None:
                raise ValueError("--compare-direct requires an Ollama model.")
            detail["direct_answer"] = ollama.answer_without_evidence(row["query"])
        detail_rows.append(detail)

    answerability = _binary_metrics(gold_answerability, predicted_answerability)
    summary: dict[str, Any] = {
        "answerability": answerability,
        "answer_quality": {
            "exact_match": sum(em_values) / max(1, len(em_values)),
            "token_f1": sum(f1_values) / max(1, len(f1_values)),
            "answerable_examples": len(em_values),
        },
        "retrieval": _retrieval_metrics(rows, hit_rows, top_k),
        "examples": len(rows),
    }

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, detail in enumerate(detail_rows):
        grouped[detail["query_language"]].append(index)
    by_language = {}
    for language, indices in grouped.items():
        language_rows = [rows[index] for index in indices]
        language_hits = [hit_rows[index] for index in indices]
        language_answerable = [
            index for index in indices if detail_rows[index]["gold_answerable"]
        ]
        by_language[language] = {
            "answerability": _binary_metrics(
                [gold_answerability[index] for index in indices],
                [predicted_answerability[index] for index in indices],
            ),
            "retrieval": _retrieval_metrics(language_rows, language_hits, top_k),
            "answer_quality": {
                "exact_match": sum(
                    float(
                        normalize_answer(detail_rows[index]["answer"])
                        == normalize_answer(detail_rows[index]["gold_answer"])
                    )
                    for index in language_answerable
                ) / max(1, len(language_answerable)),
                "token_f1": sum(
                    token_f1(
                        detail_rows[index]["answer"],
                        detail_rows[index]["gold_answer"],
                    )
                    for index in language_answerable
                ) / max(1, len(language_answerable)),
                "answerable_examples": len(language_answerable),
            },
        }
    summary["by_language"] = by_language

    if compare_direct:
        direct_em = []
        direct_f1 = []
        for detail in detail_rows:
            if detail["gold_answerable"]:
                direct_em.append(
                    float(
                        normalize_answer(detail["direct_answer"])
                        == normalize_answer(detail["gold_answer"])
                    )
                )
                direct_f1.append(token_f1(detail["direct_answer"], detail["gold_answer"]))
        summary["direct_without_retrieval"] = {
            "exact_match": sum(direct_em) / max(1, len(direct_em)),
            "token_f1": sum(direct_f1) / max(1, len(direct_f1)),
            "answerable_examples": len(direct_em),
        }
    return summary, detail_rows


def evaluate_dataset(
    passages: list[Passage],
    qa_path: str | Path,
    *,
    method: RetrievalMethod = "hybrid",
    split: str = "test",
    top_k: int = 5,
    threshold: float | None = None,
    calibrate_on_dev: bool = True,
    embedding_model: str = "sentence-transformers/LaBSE",
    alpha: float = 0.55,
    ollama: OllamaChatGenerator | None = None,
    compare_direct: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = load_qa(qa_path)
    retriever = HybridRetriever(passages, embedding_model=embedding_model, alpha=alpha)
    dev_rows = [row for row in rows if row.get("split") == "dev"]
    selected_threshold = threshold
    if selected_threshold is None:
        if calibrate_on_dev and split != "dev":
            selected_threshold = calibrate_threshold(
                retriever, dev_rows, method=method, top_k=top_k
            )
        else:
            selected_threshold = 0.42
    selected_rows = [row for row in rows if split == "all" or row.get("split") == split]
    if not selected_rows:
        raise ValueError(f"No QA rows found for split={split!r}.")
    summary, details = _score_rows(
        retriever,
        selected_rows,
        method=method,
        top_k=top_k,
        threshold=selected_threshold,
        ollama=ollama,
        compare_direct=compare_direct,
    )
    summary.update(
        {
            "split": split,
            "retrieval_method": method,
            "top_k": top_k,
            "threshold": selected_threshold,
            "threshold_calibrated_on": "dev" if calibrate_on_dev and split != "dev" and threshold is None else None,
            "embedding_model": embedding_model if method in {"dense", "hybrid"} else None,
            "hybrid_alpha": alpha if method == "hybrid" else None,
        }
    )
    hard_cases = [row for row in details if row["hard_case"]]
    summary["hard_case_examples"] = len(hard_cases)
    summary["hard_case_answerability"] = _binary_metrics(
        [row["gold_answerable"] for row in hard_cases],
        [row["predicted_answerable"] for row in hard_cases],
    ) if hard_cases else {}
    return summary, details


def write_evaluation(
    summary: dict[str, Any],
    details: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    stem: str = "evaluation",
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if details:
        with (destination / f"{stem}_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(details[0]))
            writer.writeheader()
            writer.writerows(details)
