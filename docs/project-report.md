# IndicRAG-QA: Cross-Lingual Retrieval and Evidence-Grounded Question Answering for Indian Education Policy

**Project report**

**Prepared:** 19 November 2026

**Repository:** `nlp-cross-lingual-project`

## Abstract

IndicRAG-QA is a research demonstration for answering questions about India’s National Education Policy (NEP) 2020 in English, Hindi, and Romanized Hindi-English (Hinglish). It compares lexical, character n-gram, multilingual dense, and hybrid passage retrieval. A score threshold controls whether the system answers or abstains; answers include passages and source links so a reader can inspect the supporting text. The default answerer extracts a sentence from retrieved evidence. An optional local Ollama model can generate a response from the same evidence.

The repository contains a 30-passage bilingual starter corpus and 45 annotated questions (12 development, 33 test). The recorded benchmark evaluates the starter corpus only. In November 2026, the local extended corpus was built from full official NEP texts in English and Hindi and UDISE+ NEP-structure reports for 2024–25 and 2025–26. This adds 2,965 passages, for 2,995 searchable passages across the starter and extended collections. The extension has not yet been evaluated against a newly annotated test set.

On the existing 33-question test split, LaBSE dense retrieval achieved the highest aggregate Recall@5 (0.896) and Precision@5 (0.358). Character TF-IDF achieved the highest answerability F1 (1.000) and token F1 (0.615). These results are useful as a software baseline, but the test set is small and the passage collection used for those results is the starter corpus. They should not be interpreted as broad evidence of policy-answering performance.

## 1. Problem and objectives

Education policy information is published in long documents and is often requested in more than one language. A useful question-answering system must find relevant evidence when the question and source passage use different languages or writing systems. It must also distinguish “the corpus does not answer this” from a weak but superficially similar match.

This project has four objectives:

1. Compare lexical, character-based, multilingual dense, and hybrid retrieval on English, Hindi, and code-mixed questions.
2. Return short answers with source passages that people can inspect.
3. Abstain when the top retrieval score is below a selected threshold.
4. Provide an interactive interface for asking questions, comparing retrievers, inspecting corpus passages, and adding source documents.

The project is a research prototype. It is not a complete policy archive, legal interpretation service, or production system.

### Research questions

- **RQ1:** How well do lexical, character n-gram, and multilingual dense retrieval recover annotated evidence for English, Hindi, and Hinglish questions?
- **RQ2:** Does the configured hybrid score improve on either retrieval family on this small evaluation set?
- **RQ3:** How does a development-calibrated score threshold affect answerability decisions and answer quality?
- **RQ4:** Can citations and a corpus explorer make retrieval behavior easier to inspect and diagnose?

## 2. Background

The system uses pretrained models rather than training a Transformer from scratch. The original Transformer paper supplies the architectural background for modern attention-based encoders. XLM-R motivates multilingual pretraining and evaluation by language, while the project uses LaBSE sentence embeddings for cross-language semantic retrieval. The LaBSE work studies multilingual sentence embeddings and bilingual text retrieval. Research on massively multilingual transfer also motivates caution about assuming that transfer quality is uniform across languages; that work studies named entity recognition, so it is background rather than direct evidence for this QA task. [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762), [Conneau et al. (2020)](https://aclanthology.org/2020.acl-main.747/), [Feng et al. (2022)](https://aclanthology.org/2022.acl-long.62/), [Rahimi et al. (2019)](https://aclanthology.org/P19-1015/).

## 3. Data and corpus

### Starter data

The versioned starter corpus has 30 passages: 15 English and 15 Hindi. It is made from curated NEP summaries and is designed for quick use. The QA file has 45 questions balanced across three query-language labels (15 English, 15 Hindi, and 15 code-mixed). It has 12 development questions and 33 test questions. The dev split has six answerable and six unanswerable questions; the test split has 24 answerable and nine unanswerable questions. Each split has equal numbers of the three language labels. There are 32 questions marked as hard cases.

Questions include reference answers, answerability labels, and expected evidence passage IDs. For answerable questions, the gold evidence can include both the English and Hindi passages for the same policy point. Test performance therefore depends on the starter passage IDs and corpus snapshot.

### Expanded official corpus

The optional expanded corpus adds the following text-based PDF sources:

| Document | Language | Indexed passages |
| --- | --- | ---: |
| Full NEP 2020 text | English | 562 |
| Full NEP 2020 text | Hindi | 761 |
| UDISE+ Report 2024–25, NEP structure | English | 826 |
| UDISE+ Report 2025–26, NEP structure | English | 816 |
| **Extended corpus total** | English and Hindi | **2,965** |

Together with the 30 starter passages, the selected expanded corpus contains 2,995 passages across six document IDs. The full NEP texts are hosted by SCERT Delhi; the UDISE+ reports are published by the Department of School Education & Literacy, Ministry of Education. Source URLs and publisher fields are maintained in [`data/sources.json`](../data/sources.json). The official UDISE+ 2025–26 report was released on 7 July 2026. [SCERT Delhi NEP documents](https://scert.delhi.gov.in/scert/nep-2020-documents), [Ministry of Education UDISE+ release](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2282141&lang=2&reg=48).

The build script downloads source PDFs into Git-ignored `data/raw/`, extracts text with `pypdf`, segments it into 900-character passages with 120-character overlap, and writes JSONL to Git-ignored `data/processed/passages.jsonl`. It records a source URL with every passage. The starter corpus remains versioned; the generated extended corpus is local and must be rebuilt after cloning.

The extended corpus broadens coverage from NEP text to official school statistics. Its passages have not undergone a separate human audit, and the existing QA annotations do not cover these new document IDs. Evaluation figures in this report therefore refer to the starter collection, not the 2,995-passage expanded collection.

## 4. System methods

### Ingestion and passage preparation

For PDFs, the ingestion path extracts text page by page. Text is Unicode-normalized, control characters and excess spaces are cleaned, and blocks are split at blank lines or detected headings. Long blocks are divided into overlapping chunks. Language labels are assigned with a lightweight script heuristic: Devanagari indicates Hindi, Latin text indicates English, and mixed scripts indicate code-mixed text. This label is metadata; retrieval itself is not restricted by the detected query language.

### Retrieval

All methods rank the same passage collection and return the top *K* passages.

- **BM25:** token-based lexical matching with document-length normalization. The raw score is mapped to a bounded confidence-like score using `1 − exp(−score / 4)`.
- **Character TF-IDF:** cosine similarity over character n-grams of length 2–5. Character features can tolerate some spelling and word-form variation.
- **LaBSE dense:** multilingual sentence embeddings are normalized and compared with cosine similarity. Dense passage embeddings are created lazily and cached in memory.
- **Hybrid:** combines the strongest lexical signal (the maximum of normalized BM25 and character TF-IDF) with dense similarity: `alpha × lexical + (1 − alpha) × dense`. The default `alpha` is 0.55.

### Answering and abstention

The system compares the best hit score with a threshold. If the score is below the threshold, the response abstains. Otherwise, the default answerer selects an evidence sentence from retrieved passages. The optional Ollama generator receives retrieved passages and is instructed to answer only from that evidence. The response carries the retrieved excerpts, passage IDs, document metadata, scores, and source links. Scores are ranking signals, not calibrated probabilities.

### Interactive application

The Streamlit application has four main views: **Ask**, **Compare methods**, **Explore corpus**, and **About**. Users can set the retrieval method and answerability threshold; compare rankings for a question; filter corpus passages by language, document, or text; and upload a text-based PDF, TXT, or Markdown document for local indexing. The expanded corpus can be enabled or disabled in the sidebar.

## 5. Experimental setup

The benchmark snapshot uses the 30-passage starter corpus, 33 held-out test questions, and top 5 retrieval results. The answerability threshold for each method was selected by F1 on the 12-row development split and applied unchanged to test. Retrieval metrics are calculated on 24 answerable test questions. Answerability metrics use all 33 questions. Exact match and token F1 are calculated over the 24 answerable questions. Language-specific retrieval results have eight answerable test examples per language.

The recorded report does not include hardware, elapsed latency, random seeds, or complete package lock data. Dense or hybrid use requires LaBSE model weights to be available locally or downloaded on first use. No scores for the expanded corpus or the optional Ollama answerer are included in the committed benchmark snapshot.

## 6. Results

### Aggregate test results

| Method | Dev threshold | Answerability accuracy | Answerability F1 | Exact match | Token F1 | Recall@5 | Precision@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.96 | 0.848 | 0.894 | 0.125 | 0.535 | 0.500 | 0.200 | 1.000 |
| Character TF-IDF | 0.25 | **1.000** | **1.000** | 0.167 | **0.615** | 0.500 | 0.200 | 1.000 |
| LaBSE dense | 0.47 | 0.697 | 0.773 | 0.083 | 0.444 | **0.896** | **0.358** | 0.979 |
| Hybrid (`alpha = 0.55`) | 0.73 | 0.758 | 0.810 | 0.083 | 0.484 | 0.604 | 0.242 | 1.000 |

### Recall@5 by query language

| Method | English | Hindi | Code-mixed |
| --- | ---: | ---: | ---: |
| BM25 | 0.500 | 0.500 | 0.500 |
| Character TF-IDF | 0.500 | 0.500 | 0.500 |
| LaBSE dense | **0.938** | **0.750** | **1.000** |
| Hybrid (`alpha = 0.55`) | 0.500 | 0.500 | 0.813 |

These values reproduce the committed [benchmark snapshot](benchmark-results.md). They are not measurements from the expanded corpus.

### Interpretation

Dense retrieval has substantially higher passage coverage on this test split, including the code-mixed slice. Character TF-IDF has the strongest answerability and token-overlap results in this run. The configured hybrid method does not outperform dense retrieval on aggregate recall or precision. This indicates that retrieval coverage and answerability gating should be reported separately: a method can retrieve the right evidence somewhere in the top five yet still select an unsuitable rank-one passage or abstain incorrectly.

The perfect answerability F1 for TF-IDF is an observation over only 33 test examples. The language slices contain eight answerable test questions apiece, so one changed example could materially alter a slice score. The threshold values also differ by method; they are development-set cutoffs, not probabilities that an answer is correct.

## 7. Error analysis

The repository includes first-pass review of 16 flagged test cases. It is engineering diagnosis, not independent domain-expert adjudication. Findings include:

- Dense and hybrid retrieval sometimes abstain on Romanized Hindi-English questions even when a gold passage is retrieved in the top five.
- Dense ranking can place a related but incorrect passage first; on a numeric question, this led to an unsupported answer despite the correct evidence being in the top five.
- Dense retrieval and, in one case, BM25 falsely answered scholarship questions outside the starter corpus.
- Some extractive answers contain more detail than the reference answer, which lowers exact match despite partial semantic correctness.

The highest-priority improvements are to expand and independently review the QA set, add more Romanized Hindi and near-neighbor hard negatives, calibrate answerability with a larger development split, and assess whether the proposed answer is entailed by its supporting sentence. See [error-analysis.md](error-analysis.md) for case-level notes.

## 8. Limitations, ethics, and data handling

1. **Small test set:** 33 test examples are insufficient for broad performance claims; per-language slices are especially small.
2. **Evidence annotations:** the QA set was curated for the starter passages and needs review before reuse with the added full-text and statistics corpus.
3. **PDF extraction:** table-heavy reports and complex page layouts may not extract cleanly. OCR is not included for scanned pages.
4. **Language coverage:** the corpus is English and Hindi. Hinglish is represented in queries, not as a standalone source-language corpus. The language detector is heuristic.
5. **Answer reliability:** retrieval score thresholds do not prove entailment or factual correctness. Extractive selection can choose a sentence related to the question but fail to answer it.
6. **Source scope and freshness:** the corpus contains specific NEP and UDISE+ snapshots, not every official policy or the complete live education data system. Verify time-sensitive statistics against the source report.
7. **Data governance:** source URLs and publishers are recorded; redistribution and reuse terms should be checked for each source before republishing source PDFs or derived data.
8. **Uploaded documents:** the app writes uploaded documents under `data/raw/` and extracted passages under `data/processed/`. Those folders are Git-ignored. For a shared deployment, access control, retention, and deletion procedures would need to be added.
9. **Optional generation:** Ollama is a local-model option; generation can still add unsupported details. Answers should be checked against the cited passages.

## 9. Conclusion and next steps

IndicRAG-QA provides a compact, inspectable testbed for cross-lingual retrieval over education-policy material. Its current benchmark suggests that dense LaBSE retrieval improves evidence recall on the starter set, while the character TF-IDF baseline performs better on answerability and answer token overlap. The expanded source collection makes the prototype more useful for exploration, but it does not yet have a matching reviewed evaluation set.

Recommended next steps:

1. Review extraction quality, page numbers, headings, and chunks for all four added PDFs.
2. Add and double-review 300–500 questions with stable evidence IDs for the expanded corpus; split by document or policy section where feasible.
3. Keep a development split for threshold and hybrid-weight selection; freeze the new test split before evaluation.
4. Report metrics by language, answerability, and source document, with confidence intervals where feasible.
5. Add sentence-level entailment or evidence verification for numeric claims and out-of-domain questions.
6. Record model, library, hardware, latency, download date, and source checksums in each experiment output.

## 10. Reproduction

From the repository root, activate the project environment, install `requirements.txt`, and run the app with `streamlit run app.py`. To rebuild the expanded corpus and source files, run:

```bash
python scripts/build_extended_corpus.py
```

The benchmark commands and generated outputs are documented in [benchmark-results.md](benchmark-results.md). Use BM25 or character TF-IDF for a first run without downloading LaBSE weights. Use the expanded-corpus toggle to search the locally generated full-text passages.

## References

### Research background

- Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Conneau, A. et al. (2020). [Unsupervised Cross-lingual Representation Learning at Scale](https://aclanthology.org/2020.acl-main.747/).
- Feng, F. et al. (2022). [Language-agnostic BERT Sentence Embedding](https://aclanthology.org/2022.acl-long.62/).
- Rahimi, A., Li, Y., & Cohn, T. (2019). [Massively Multilingual Transfer for NER](https://aclanthology.org/P19-1015/).

### Data sources

- Ministry of Education, Government of India. [National Education Policy 2020 documents](https://scert.delhi.gov.in/scert/nep-2020-documents), English and Hindi editions hosted by SCERT Delhi.
- Department of School Education & Literacy, Ministry of Education. [UDISE+ 2025–26 report release](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2282141&lang=2&reg=48); the 2024–25 and 2025–26 NEP-structure report PDFs are linked in [`data/sources.json`](../data/sources.json).
