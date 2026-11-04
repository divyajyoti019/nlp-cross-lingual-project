# IndicRAG-QA benchmark snapshot

These are the bundled software-demo results from the 33-row held-out test split. Thresholds were selected on the 12-row dev split and then applied unchanged to test. Retrieval uses top 5 passages. Answer-quality scores are measured on the 24 answerable test examples; retrieval scores are measured on those same 24 examples. Answerability scores use all 33 test examples.

## Aggregate results

| Method | Dev threshold | Answerability accuracy | Answerability F1 | Exact match | Token F1 | Recall@5 | Precision@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.96 | 0.848 | 0.894 | 0.125 | 0.535 | 0.500 | 0.200 | 1.000 |
| Character TF-IDF | 0.25 | 1.000 | 1.000 | 0.167 | 0.615 | 0.500 | 0.200 | 1.000 |
| LaBSE dense | 0.47 | 0.697 | 0.773 | 0.083 | 0.444 | **0.896** | **0.358** | 0.979 |
| Hybrid (alpha 0.55) | 0.73 | 0.758 | 0.810 | 0.083 | 0.484 | 0.604 | 0.242 | 1.000 |

Precision@5 is relevant passages divided by five, as defined in the project evaluation protocol. For each answerable question, gold annotations may include both English and Hindi source passages.

## Retrieval by query language

| Method | English Recall@5 | Hindi Recall@5 | Code-mixed Recall@5 |
| --- | ---: | ---: | ---: |
| BM25 | 0.500 | 0.500 | 0.500 |
| Character TF-IDF | 0.500 | 0.500 | 0.500 |
| LaBSE dense | **0.938** | **0.750** | **1.000** |
| Hybrid (alpha 0.55) | 0.500 | 0.500 | 0.813 |

Each language slice has only eight answerable test questions, so a single changed example can move a slice substantially.

## Reading these results

- Dense LaBSE retrieval has the highest aggregate Recall@5 (0.896) and Precision@5 (0.358) in this run. It improves passage coverage across all three language slices, while its answerability F1 and answer token F1 are lower than the lexical baselines.
- Character TF-IDF has the strongest answerability and answer-overlap scores on this split. The perfect answerability score is an observation on 33 examples, not evidence of general reliability.
- The hybrid configuration does not beat dense retrieval on aggregate passage coverage in this run. Its weighting should be tuned on development data only; the test set must remain fixed for comparison.
- MRR@5 is near 1.0 because, when a gold passage is retrieved, it is usually ranked first. This does not mean every relevant passage is found; Recall@5 shows the missed cases.

## Scope and next work

This snapshot is a reproducible software baseline, not a research-grade benchmark. The 45-row dataset contains 12 dev rows and 33 test rows; the language-specific retrieval figures use eight answerable test questions per language. The corpus contains curated NEP summaries rather than a full, independently chunked set of source documents. The dataset includes gold passage IDs and answer references, but the current test outcomes and hard-case error records have not received a separate human adjudication pass.

Before making research claims, ingest the full source documents, grow the dataset toward the assignment's 300–500 reviewed questions, and double-check the passage and answer annotations. A first-pass review of 16 flagged test cases is documented in [error-analysis.md](error-analysis.md); it should be followed by independent subject-matter review. Keep thresholds and any retrieval weights selected on dev; do not tune them against this test split. See [the evaluation protocol](evaluation.md) and [the data card](data-card.md) for the annotation and provenance details.

## Reproduce

Run from the repository root with the project environment activated. The LaBSE weights are downloaded from Hugging Face on first use; if they are already cached, offline mode can be used.

```bash
PYTHONPATH=src python -m indicrag.cli evaluate --method bm25 --split test
PYTHONPATH=src python -m indicrag.cli evaluate --method tfidf --split test
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m indicrag.cli evaluate --method dense --split test
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m indicrag.cli evaluate --method hybrid --split test
```

The CLI writes per-example CSV predictions and JSON summaries to `outputs/evaluation/`. Those generated outputs are ignored by Git; this committed snapshot records the headline metrics for review.
