# Architecture

## Request path

1. A user submits an English, Hindi, or code-mixed question.
2. The query-type heuristic reports en, hi, or code_mixed; retrieval itself is not limited by that label.
3. A selected retriever ranks the same passage collection:
   - BM25: lexical term-matching baseline.
   - TF-IDF: character n-gram cosine baseline, useful for spelling and morphology variation.
   - LaBSE: normalized multilingual sentence embeddings and cosine similarity.
   - Hybrid: alpha * lexical_score + (1 - alpha) * dense_score.
4. A threshold calibrated on development examples decides whether evidence is sufficient.
5. If sufficient, the baseline answerer selects a sentence from the retrieved passages. An optional Ollama model can generate a concise answer using only those passages.
6. The response returns the answerability decision, score, source passage IDs, document URL, section, and excerpt.

## Why the answerer is extractive by default

An extractive answer makes the evidence trail inspectable and gives the project a no-key, repeatable baseline. A generative model is an optional ablation, not an assumed source of truth. The LLM prompt treats source text as untrusted quoted evidence and asks it to abstain if the evidence does not answer the question.

## Data flow

- data/sample/passages.jsonl: versioned seed passages for immediate use.
- data/raw/: user-provided source documents; ignored by Git.
- data/processed/passages.jsonl: generated passage collection; ignored by Git.
- data/eval/qa.jsonl: manually reviewable development/test examples with evidence IDs.
- outputs/: experiment summaries and row-level predictions; ignored by Git.

## Research-paper mapping

The Transformer paper motivates attention-based encoders. XLM-R motivates multilingual pretraining and careful capacity/language reporting. LaBSE is the retrieval encoder because its training objective places translations in a shared sentence space. The multilingual transfer paper motivates language-specific error analysis rather than assuming one transfer setup works equally for every language.
