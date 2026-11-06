"""Command line entry point for IndicRAG-QA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .evaluation import evaluate_dataset, write_evaluation
from .generation import OllamaChatGenerator
from .ingest import ingest_file, load_passages, write_passages
from .pipeline import IndicRAG
from .retrieval import DEFAULT_EMBEDDING_MODEL, RetrievalMethod

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PASSAGES = ROOT / "data" / "sample" / "passages.jsonl"
EXTRA_PASSAGES = ROOT / "data" / "processed" / "passages.jsonl"
DEFAULT_QA = ROOT / "data" / "eval" / "qa.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "evaluation"


def _load_project_passages(path: Path | None = None):
    passages = load_passages(path or SAMPLE_PASSAGES)
    if path is None and EXTRA_PASSAGES.exists():
        passages.extend(load_passages(EXTRA_PASSAGES))
    return passages


def _generator(enabled: bool) -> OllamaChatGenerator | None:
    if not enabled:
        return None
    return OllamaChatGenerator(os.getenv("OLLAMA_MODEL"))


def _print_result(result: dict) -> None:
    print(f"Query language: {result['query_language']}")
    print(f"Answerable: {'ANSWERABLE' if result['answerable'] else 'UNANSWERABLE'}")
    print(f"Confidence: {result['confidence']:.3f} (threshold {result['threshold']:.3f})")
    print(f"\n{result['answer']}\n")
    if result["citations"]:
        print("Evidence:")
        for citation in result["citations"]:
            section = f", section {citation['section']}" if citation["section"] else ""
            page = f", page {citation['page']}" if citation["page"] else ""
            print(
                f"- {citation['passage_id']} — {citation['title']}{section}{page} "
                f"(score {citation['score']:.3f})"
            )
            if citation["source_url"]:
                print(f"  Source: {citation['source_url']}")


def _add_common_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--alpha", type=float, default=0.55, help="Lexical weight in hybrid retrieval.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="indicrag",
        description="Evidence-grounded English, Hindi, and code-mixed question answering.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="Ask a question against the indexed passage collection.")
    ask.add_argument("query", nargs="+")
    ask.add_argument("--method", choices=["bm25", "tfidf", "dense", "hybrid"], default="hybrid")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--threshold", type=float, default=0.42)
    ask.add_argument("--passages", type=Path, help="Use one JSONL passage file instead of the project corpus.")
    ask.add_argument("--ollama", action="store_true", help="Generate a concise answer with local Ollama.")
    _add_common_model_options(ask)

    ingest = commands.add_parser("ingest", help="Extract passages from PDF, TXT, or Markdown.")
    ingest.add_argument("input", type=Path)
    ingest.add_argument("--output", type=Path, default=EXTRA_PASSAGES)
    ingest.add_argument("--doc-id")
    ingest.add_argument("--title")
    ingest.add_argument("--source-url", default="")
    ingest.add_argument("--language", default="auto", choices=["auto", "en", "hi", "code_mixed"])
    ingest.add_argument("--max-chars", type=int, default=900)
    ingest.add_argument("--overlap-chars", type=int, default=120)

    evaluate = commands.add_parser("evaluate", help="Score retrieval, answers, and abstention behavior.")
    evaluate.add_argument("--dataset", type=Path, default=DEFAULT_QA)
    evaluate.add_argument("--split", choices=["dev", "test", "all"], default="test")
    evaluate.add_argument("--method", choices=["bm25", "tfidf", "dense", "hybrid"], default="hybrid")
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--threshold", type=float)
    evaluate.add_argument("--no-calibration", action="store_true")
    evaluate.add_argument("--ollama", action="store_true")
    evaluate.add_argument("--compare-direct", action="store_true")
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    _add_common_model_options(evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        files = sorted(args.input.rglob("*")) if args.input.is_dir() else [args.input]
        supported = [path for path in files if path.suffix.lower() in {".pdf", ".txt", ".md"}]
        if not supported:
            raise SystemExit("No PDF, TXT, or Markdown files found.")
        passages = []
        for path in supported:
            passages.extend(
                ingest_file(
                    path,
                    doc_id=args.doc_id if len(supported) == 1 else None,
                    title=args.title if len(supported) == 1 else None,
                    source_url=args.source_url,
                    language=args.language,
                    max_chars=args.max_chars,
                    overlap_chars=args.overlap_chars,
                )
            )
        write_passages(passages, args.output)
        print(f"Wrote {len(passages)} passages to {args.output.resolve()}")
        return 0

    passages = _load_project_passages(args.passages if args.command == "ask" else None)
    if args.command == "ask":
        generator = _generator(args.ollama)
        app = IndicRAG(
            passages,
            embedding_model=args.embedding_model,
            alpha=args.alpha,
            ollama=generator,
        )
        result = app.ask(
            " ".join(args.query),
            method=args.method,
            top_k=args.top_k,
            threshold=args.threshold,
            use_ollama=args.ollama,
        )
        _print_result(result)
        return 0

    generator = _generator(args.ollama or args.compare_direct)
    summary, details = evaluate_dataset(
        passages,
        args.dataset,
        method=args.method,
        split=args.split,
        top_k=args.top_k,
        threshold=args.threshold,
        calibrate_on_dev=not args.no_calibration,
        embedding_model=args.embedding_model,
        alpha=args.alpha,
        ollama=generator if (args.ollama or args.compare_direct) else None,
        compare_direct=args.compare_direct,
    )
    write_evaluation(summary, details, args.output_dir, stem=f"{args.method}_{args.split}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDetailed predictions: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
