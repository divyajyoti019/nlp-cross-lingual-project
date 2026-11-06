# Data card

## Starter collection

- **Domain:** Indian school education policy.
- **Sources:** National Education Policy 2020 in English and Hindi; URLs and publisher are listed in data/sources.json.
- **Bundled rows:** 24 curated short passage summaries, plus 45 question examples split into development and test sets.
- **Languages:** English, Hindi, and Romanized Hindi-English code-mixed queries.
- **Labels:** answerability, gold answer, expected evidence passage IDs, language slice, and difficult-case flag.
- **Limitations:** short starter collection, manually summarized policy facts, and heuristic code-mixed language tagging. It does not represent all Indian languages, regional spelling variants, or the full NEP document.

## Before research use

1. Download full source documents from the listed official URLs and record the retrieval date and SHA-256 checksum.
2. Ingest full documents and inspect extracted text, page numbers, headings, and passages.
3. Add a second person to review each question, answer, language label, answerability label, and evidence ID.
4. Expand to the assignment's recommended 300-500 question-answer examples.
5. Split by source document or policy section where feasible to reduce near-duplicate leakage.
6. Keep the original documents and any licensing or reuse notes with the dataset.

Do not use generated paraphrases as gold labels without human review. Synthetic unanswerable examples should be checked to ensure the answer is absent from every indexed source.
