# Data card

## Starter collection

- **Domain:** Indian school education policy.
- **Sources:** National Education Policy 2020 in English and Hindi; URLs and publisher are listed in data/sources.json.
- **Bundled passages:** 30 curated short passage summaries (15 English and 15 Hindi).
- **Question set:** 45 curated, annotated examples: 12 development and 33 test. The set is balanced across English, Hindi, and code-mixed queries (15 each); 32 examples are marked as hard cases. The annotations have not received an independent domain-expert adjudication pass.
- **Languages:** English, Hindi, and Romanized Hindi-English code-mixed queries.
- **Labels:** answerability, gold answer, expected evidence passage IDs, language slice, and difficult-case flag.
- **Limitations:** short starter collection, manually summarized policy facts, and heuristic code-mixed language tagging. It does not represent all Indian languages, regional spelling variants, or the full NEP document.

## Optional expanded corpus

The local build script adds 2,965 passages from the full NEP 2020 text in English and Hindi, and UDISE+ NEP-structure reports for 2024–25 and 2025–26. Combined with the starter corpus, this is 2,995 passages across six document IDs. Source URLs and publishers are recorded in `data/sources.json`; PDFs are downloaded into the ignored `data/raw/` folder and generated passages into the ignored `data/processed/` folder.

The current 45-question annotation set is tied to the starter passage IDs. It has not been expanded or re-reviewed for the optional documents. Do not treat benchmark results on the starter set as an evaluation of the expanded corpus.

## Before research use

1. Record retrieval dates and SHA-256 checksums for source PDFs used in an experiment.
2. Inspect extracted text, page numbers, headings, table text, and passages, especially in the UDISE+ reports.
3. Add a second reviewer to check each question, answer, language label, answerability label, and evidence ID.
4. Expand to the assignment's recommended 300–500 question-answer examples for the expanded corpus.
5. Split by source document or policy section where feasible to reduce near-duplicate leakage.
6. Keep the original documents and any licensing or reuse notes with the dataset.

Do not use generated paraphrases as gold labels without human review. Synthetic unanswerable examples should be checked to ensure the answer is absent from every indexed source.
