# Difficult-case review

This is a first-pass inspection of 16 flagged examples from the 33-row test split. I compared each gold evidence ID with the top-five predictions and checked the answer or abstention emitted by BM25, character TF-IDF, LaBSE dense, and hybrid retrieval. This is useful for engineering diagnosis; it is not an independent domain-expert validation of NEP content or annotations.

| ID | Query type and focus | Retrieval finding | Answer / abstention finding | Proposed change |
| --- | --- | --- | --- | --- |
| test-001 | English; preferred mother-tongue grade | Gold section 4.11 appears in top five for all four methods. | All answer; extractive answer includes the Grade 5 minimum and Grade 8 preference. | Keep this as a positive control for numeric and qualification handling. |
| test-003 | Romanized Hindi-English; mother-tongue grade | Gold section 4.11 appears in top five for all methods. | Dense and hybrid abstain despite retrieving the evidence; BM25 and TF-IDF answer. | Add more Romanized variants to dev data and calibrate answerability thresholds by language on a larger dev set. |
| test-006 | Romanized Hindi-English; whether a language is imposed on a State | Gold section 4.13 appears in top five for all methods. | Dense and hybrid abstain. BM25 and TF-IDF return a sentence about flexibility that does not directly answer “no language is imposed.” | Select and verify evidence at sentence level; require the answer sentence to entail the requested claim. |
| test-009 | Romanized Hindi-English; stage when coding starts | Gold section 4.25 appears in top five for all methods. | Dense and hybrid abstain; BM25 and TF-IDF give the Middle Stage answer. | Add code-mixed dev examples and use a query-aware answerability calibration. |
| test-010 | English; duration of the Grades 6–8 bagless period | Gold section 4.26 is in the dense top five, but dense ranks section 4.16 first. Other methods rank the gold section first. | Dense answers with an unrelated Languages of India project sentence; other methods return the 10-day period. | Rerank the candidate passages before answer extraction and test rank-one evidence for numeric questions. |
| test-012 | Romanized Hindi-English; bagless-period duration | Gold section 4.26 appears in top five for all methods. | All methods answer with the 10-day period. | Keep as a positive control; add spelling variants around “bagless” and grade ranges. |
| test-013 | English; 2025 vocational-exposure percentage | Gold section 16.5 appears in top five for all methods. | All methods answer at least 50%. | Preserve the year-and-percentage check in regression review. |
| test-018 | Romanized Hindi-English; arts/science separation | Gold section 4.9 appears in top five for all methods. | TF-IDF answers correctly; BM25, dense, and hybrid abstain. | Calibrate code-mixed thresholds on more examples and evaluate answerability separately from retrieval hit rate. |
| test-021 | Romanized Hindi-English; bilingual science and mathematics textbooks | Gold section 4.14 appears in top five for all methods. | TF-IDF and dense answer; BM25 and hybrid abstain. The extractive answer is longer than the short gold answer but contains the requested subjects and bilingual-material detail. | Review semantic answer scoring and use sentence-level evidence selection for short list questions. |
| test-024 | Romanized Hindi-English; holistic progress card | Gold section 4.35 appears in top five for all methods. | All methods answer with the holistic, 360-degree report. | Keep as a positive control for code-mixed questions. |
| test-025 | English; scholarship application deadline (unanswerable) | No gold passage exists; dense retrieves a vocational-exposure passage. | Dense gives an unrelated answer containing a year and percentage; the other three methods abstain. | Add hard negatives with overlapping time/amount terms and require evidence entailment before answering. |
| test-026 | Hindi; scholarship application deadline (unanswerable) | No gold passage exists; BM25 and dense retrieve unrelated policy sections. | BM25 and dense return unrelated policy claims; TF-IDF and hybrid abstain. | Add Hindi hard negatives and calibrate Hindi answerability separately after expanding the dev set. |
| test-027 | Romanized Hindi-English; scholarship annual fee (unanswerable) | No gold passage exists; retrieved policy sections are unrelated. | All four methods abstain. | Retain as an abstention control and add paraphrases to ensure the behavior is stable. |
| test-029 | Hindi; hostel allowance (unanswerable) | No gold passage exists; retrieved policy sections are unrelated. | All four methods abstain. | Retain as a Hindi abstention control. |
| test-030 | Romanized Hindi-English; scholarship portal OTR instructions (unanswerable) | No gold passage exists; retrieved policy sections are unrelated. | All four methods abstain. | Retain as a code-mixed abstention control; include portal-specific distractors in future evaluation. |
| test-031 | English; scholarship family-income ceiling (unanswerable) | No gold passage exists; retrieved policy sections are unrelated. | All four methods abstain. | Retain as an entity-and-amount abstention control. |

## Main patterns

- For all ten answerable examples in this sample, each method retrieves at least one gold passage in its top five. Answerability gating and answer selection therefore explain several failures even when retrieval coverage is present.
- Romanized code-mixed questions are a common false-abstention pattern for dense and hybrid retrieval. A single global threshold is not reliable enough for that slice on this small dataset.
- Dense ranking can still put an unrelated passage first even when the gold passage is in the top five, as in test-010. This can produce an unsupported extractive answer.
- Dense retrieval falsely answers two scholarship questions that the seed corpus does not cover (test-025 and test-026); BM25 also falsely answers test-026. Retrieval similarity alone is not a sufficient answerability test.
- The reviewed unanswerable examples test out-of-domain details such as scholarship deadlines, amounts, income limits, hostel allowances, and portal instructions. Preserve these hard negatives when expanding the corpus and dev set.

## Prioritized fixes

1. Expand the reviewed QA set, with more Romanized Hindi, spelling variants, and answerable/unanswerable near-neighbors. Keep the test split untouched during tuning.
2. Calibrate answerability by language on a larger dev split and record uncertainty; do not rely on one threshold for every language.
3. Add a reranking or entailment check between retrieved passage and proposed answer, especially for numbers, dates, amounts, and “is a rule imposed?” questions.
4. Inspect sentence-level extraction for passages containing several related facts, then report answer quality and abstention errors separately from retrieval recall.

The aggregate method comparison is in [benchmark-results.md](benchmark-results.md); the evaluation procedure is in [evaluation.md](evaluation.md).
