# IndicRAG-QA

Evidence-grounded cross-lingual question answering for English, Hindi, and Hindi-English code-mixed queries.

The system retrieves passages from a small bilingual education-policy corpus, answers from the retrieved evidence, cites the supporting passage, and abstains when the top evidence score is below a configurable threshold. The starter documents are English and Hindi versions of India's National Education Policy 2020.

## Research design

The attached project brief asks for separate lexical, multilingual dense, and hybrid retrieval experiments; language-stratified evaluation; answerability handling; citations; and difficult-case analysis. This repository implements those components as a reproducible baseline.

The four supplied papers inform the design in different ways:

- Vaswani et al. introduce the Transformer and attention-based sequence modeling. This project uses pretrained attention-based encoders instead of training a Transformer from scratch.
- Conneau et al. show the value and trade-offs of multilingual pretraining at scale. XLM-R motivates multilingual transfer and language-sliced reporting; the current retrieval model is LaBSE because the project needs sentence-level cross-lingual retrieval.
- Feng et al. describe LaBSE's multilingual sentence-embedding and translation-ranking approach. LaBSE is the default dense retriever for English-Hindi passage matching.
- Rahimi et al. show that transfer quality varies by source and target language. The project reports English, Hindi, and code-mixed results separately and retains difficult cases for review.

The raw XLM-R checkpoint is not used as a QA generator: it is a pretrained encoder, not an instruction-tuned answer model. If you add a reader or reranker later, fine-tune it on an appropriate multilingual QA or ranking dataset and compare it with the current baselines.

## What is included

- PDF, TXT, and Markdown ingestion with page, section, document, and source metadata.
- BM25 and character n-gram TF-IDF lexical baselines.
- LaBSE dense retrieval for cross-lingual sentence similarity.
- Weighted hybrid retrieval, with a tunable lexical weight.
- Script and Hinglish-lexicon query-type detection.
- Answerability gating, an extractive evidence-only answer baseline, and source citations.
- Optional local Ollama generation for RAG and direct-answer ablations.
- Evaluation of Recall@K, Precision@K, MRR, exact match, token F1, answerability accuracy/precision/recall/F1, language slices, and difficult cases.
- Streamlit demo and a Jupyter research walkthrough.

## Quick start on macOS

Open this folder in VS Code. The workspace recommends the Microsoft Python, Pylance, and Jupyter extensions and selects .venv as its Python interpreter.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m ipykernel install --prefix .venv --name indicrag --display-name "Python (IndicRAG-QA)"

For Windows PowerShell, activate the environment with .venv\Scripts\Activate.ps1. The project supports Python 3.11-3.14; if a machine has trouble resolving ML packages, use Python 3.11 or 3.12.

Select **Python (IndicRAG-QA)** as the kernel for notebooks. Open notebooks/01_indicrag_demo.ipynb in VS Code or run jupyter lab.

The repository includes recommended VS Code extensions for Python, Pylance, and Jupyter. To register the notebook kernel after installing requirements, run:

    python -m ipykernel install --prefix .venv --name indicrag --display-name "Python (IndicRAG-QA)"

## Run the application

The lexical baselines work without downloading a model:

    indicrag ask "What does NEP say about coding activities?" --method bm25

Dense or hybrid retrieval loads LaBSE the first time and may download model weights:

    indicrag ask "NEP ke according coding activities kaunse school stage me start hoti hain?" --method hybrid

Launch the browser demo:

    streamlit run app.py

The Streamlit app lets you choose the retrieval method, top-K, score threshold, and optional local answer generator. For evidence-sensitive use, review the returned source passage.

## Add your own documents

Put source PDFs, text files, or Markdown files in data/raw/ (this folder is ignored by Git), then ingest a document:

    indicrag ingest data/raw/your_policy.pdf --doc-id your-policy --title "Your policy title" --source-url "https://example.org/policy.pdf" --language en

The generated passage file goes to data/processed/passages.jsonl, also ignored by Git. The CLI and app load it alongside the bundled sample passages. For mixed-language documents, use --language auto.

## Optional local answer generation

The default answerer returns a sentence selected from the retrieved evidence. To compare it with a generative RAG answer, install and start Ollama, pull a multilingual instruction model you are licensed to use, and set OLLAMA_MODEL. For example, the Ollama library provides qwen2.5:3b-instruct; check the model license and hardware requirements before using it.

    export OLLAMA_MODEL=qwen2.5:3b-instruct
    indicrag ask "Which grades have the bagless period?" --method hybrid --ollama

For Windows PowerShell, set it with $env:OLLAMA_MODEL="qwen2.5:3b-instruct". No hosted API key is required. The --compare-direct evaluation option compares that model without retrieved evidence against the RAG answer.

## Evaluate

Run retrieval and answerability evaluation with the threshold selected on the development split:

    indicrag evaluate --method bm25 --split test
    indicrag evaluate --method tfidf --split test
    indicrag evaluate --method dense --split test
    indicrag evaluate --method hybrid --split test

Results are written to outputs/evaluation/ as JSON summaries and CSV predictions. To run the optional direct-model comparison:

    indicrag evaluate --method hybrid --split test --ollama --compare-direct

Never tune thresholds or prompts on the test split. Report the development-calibrated threshold and include per-language and difficult-case results. The bundled QA set has 45 curated examples for software demonstration; the assignment recommends a larger 300-500 example research set, which should be built by adding domain documents and human-reviewed questions rather than duplicating templates.

## Project layout

    app.py
    src/indicrag/         Ingestion, retrieval, answer generation, evaluation, and CLI
    data/sample/          Small bilingual starter passage collection
    data/eval/            Development and test question set
    data/sources.json     Source URLs and provenance
    notebooks/            Jupyter research walkthrough
    docs/                 Architecture, evaluation protocol, and data notes
    legacy/               Previous NLP Lab 9 solution, preserved for reference

## GitHub workflow

The project remote is [divyajyoti019/nlp-cross-lingual-project](https://github.com/divyajyoti019/nlp-cross-lingual-project). To work on a separate change, create a branch, commit your files, and push that branch:

    git switch -c feature/short-description
    git add .
    git commit -m "Describe the change"
    git push -u origin feature/short-description

Then open a pull request on GitHub and review the changed files and generated evaluation outputs before merging. Do not commit `.venv/`, source PDFs, generated passages, model weights, or evaluation outputs; the repository `.gitignore` excludes these local files.

## Limitations

- The bundled passage and QA sets are a small demonstration set, not a statistically powered benchmark.
- Romanized Hinglish language detection uses a small marker list; report it as a heuristic and manually inspect misclassified queries.
- Dense retrieval quality and answerability thresholds must be calibrated on your own held-out documents and language mix.
- Extractive responses are deliberately conservative. Optional LLM answers can still be wrong; evaluate them against the evidence and preserve citations.
- Policy passages are summaries with links to official documents. For a final research evaluation, ingest the full source PDFs and annotate evidence passages directly.

## References

1. Vaswani et al. (2017). Attention Is All You Need. NeurIPS. https://arxiv.org/abs/1706.03762
2. Conneau et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. ACL. https://aclanthology.org/2020.acl-main.747/
3. Feng et al. (2022). Language-agnostic BERT Sentence Embedding. ACL. https://aclanthology.org/2022.acl-long.62/
4. Rahimi et al. (2019). Massively Multilingual Transfer for NER. ACL. https://aclanthology.org/P19-1015/
5. Ministry of Education, Government of India. National Education Policy 2020 (English and Hindi). Source URLs are recorded in data/sources.json.
