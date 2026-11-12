# IndicRAG-QA

IndicRAG-QA is a small research and demonstration project for answering questions about Indian school education policy in English, Hindi, and Hindi-English code-mixed text. It retrieves relevant policy passages, answers from that evidence, shows source citations, and can abstain when the evidence score is too low.

The bundled corpus contains curated summaries of India's National Education Policy (NEP) 2020 in English and Hindi. It is a software demo corpus, not a complete policy archive or a research-scale benchmark.

## What happens when you ask a question?

1. The question is searched against the policy passages.
2. A retriever ranks passages using BM25, character TF-IDF, multilingual LaBSE embeddings, or a hybrid score.
3. The top score is compared with an answerability threshold.
4. If evidence is sufficient, the default answerer selects a sentence from the retrieved passages. The result includes citations and evidence excerpts. If evidence is weak, the system abstains.
5. An optional local Ollama model can generate an answer from the retrieved passages.

The default hybrid score is `alpha × lexical score + (1 − alpha) × dense score`; `alpha` is `0.55` by default. The lexical score uses the stronger of the normalized BM25 and character TF-IDF scores. LaBSE is loaded only for dense or hybrid retrieval.

## Get the project

Clone the public repository:

```bash
git clone https://github.com/divyajyoti019/nlp-cross-lingual-project.git
cd nlp-cross-lingual-project
```

## Requirements

- Python 3.11–3.14 (Python 3.12 is a good default)
- Git
- Internet access for the first LaBSE model download when using dense or hybrid retrieval
- VS Code is optional; the project also runs from a terminal

Installing the Python dependencies also installs the notebook tools. The first install may take a few minutes because PyTorch and sentence-transformers are large packages.

## Install

Run these commands from the repository root.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, use Command Prompt and run `.venv\Scripts\activate.bat`, then continue with the `python -m pip` commands above.

## Run the browser app

Start the Streamlit demo from the repository root:

```bash
streamlit run app.py
```

Streamlit prints a local URL to open in your browser. The website includes example questions in English, Hindi, and Hinglish, a guided question form, expandable source passages, retrieval settings, and separate tabs explaining the methods and project scope. In the sidebar, choose a retrieval method, number of passages, answerability threshold, and hybrid lexical weight.

For the first run without a model download, choose **bm25** or **tfidf**. The default **hybrid** method loads LaBSE the first time and downloads its weights. Dense retrieval needs an internet connection on that first run; later runs can use the cached model.

## Use the command line

Ask a question with a lexical baseline:

```bash
indicrag ask "What does NEP say about coding activities?" --method bm25
```

Use cross-lingual dense or hybrid retrieval (the first run downloads LaBSE):

```bash
indicrag ask "NEP ke according coding activities kis school stage me shuru hoti hain?" --method hybrid
indicrag ask "What does NEP say about coding activities?" --method dense
```

Available methods:

| Method | What it does | Downloads model weights? |
| --- | --- | --- |
| `bm25` | Finds passages by matching words | No |
| `tfidf` | Compares character n-grams; can help with spelling and word-form variation | No |
| `dense` | Uses multilingual LaBSE sentence embeddings to match meaning across languages | Yes, on first use |
| `hybrid` | Combines lexical matching with LaBSE similarity | Yes, on first use |

Useful options for `indicrag ask`:

```text
--top-k 5          Number of passages to retrieve
--threshold 0.42   Minimum top-passage score to answer
--alpha 0.55       Lexical weight for hybrid retrieval (0–1)
--passages FILE    Use one JSONL passage file instead of the project corpus
```

The threshold is a decision setting, not a calibrated probability. Tune it on development data before evaluating on a held-out test set.

## Evaluate the retrieval methods

Run each method on the bundled test split:

```bash
indicrag evaluate --method bm25 --split test
indicrag evaluate --method tfidf --split test
indicrag evaluate --method dense --split test
indicrag evaluate --method hybrid --split test
```

The default evaluation calibrates the answerability threshold on the development split before scoring the test split. Results are written under `outputs/evaluation/` as a JSON summary and a CSV of per-question predictions. The output directory is ignored by Git.

The evaluation set has 45 curated questions: 12 development and 33 test examples. It reports answerability, answer quality, retrieval metrics, language slices, and difficult cases. Use the development split to choose thresholds or retrieval settings; keep the test split for final comparisons. Read [the evaluation protocol](docs/evaluation.md) and [the current benchmark snapshot](docs/benchmark-results.md) before interpreting results. These small-set scores are a reproducible software baseline, not evidence of performance on a broad real-world benchmark.

## Use VS Code and Jupyter

1. Open the repository folder in VS Code.
2. Install the recommended **Python**, **Pylance**, and **Jupyter** extensions when VS Code offers them.
3. Open the Command Palette and choose **Python: Select Interpreter**. Select the Python executable inside this repository's `.venv` folder.
4. To register the notebook kernel, run:

   ```bash
   python -m ipykernel install --prefix .venv --name indicrag --display-name "Python (IndicRAG-QA)"
   ```

5. Open `notebooks/01_indicrag_demo.ipynb` and select **Python (IndicRAG-QA)** as its kernel.

You can also run JupyterLab from the activated environment and repository root:

```bash
jupyter lab
```

The notebook walks through loading passages, comparing retrieval methods, inspecting citations, and evaluating on the development split.

## Add your own policy documents

The ingestion command supports text-based PDF, TXT, and Markdown files. It extracts text, splits it into passages, and records document metadata. Scanned PDFs need OCR before ingestion; OCR is not included.

1. Put your files in `data/raw/`. That folder is ignored by Git so source documents are not committed accidentally.
2. Ingest one file (quote paths that contain spaces):

   ```bash
   indicrag ingest "data/raw/your_policy.pdf" \
     --doc-id your-policy \
     --title "Your policy title" \
     --source-url "https://example.org/your-policy.pdf" \
     --language auto
   ```

3. The default output is `data/processed/passages.jsonl`. The CLI and browser app load that file along with the bundled sample passages.

For a folder, pass the folder path to `indicrag ingest`; all supported files beneath it are processed. You can adjust passage length with `--max-chars` and `--overlap-chars`. Review the generated passages and metadata before using them in an evaluation.

Each passage is a JSON object on its own line with fields such as `passage_id`, `doc_id`, `title`, `source_url`, `language`, `section`, `text`, and `page`. Keep passage IDs stable when creating QA annotations.

## Optional: use a local Ollama model

The project works without Ollama. To try local generation, install Ollama separately, download a chat model that supports your languages, and make sure the Ollama service is running. Then set the model name in the same terminal before starting the app or CLI.

macOS or Linux:

```bash
export OLLAMA_MODEL="your-installed-model-name"
export OLLAMA_BASE_URL="http://localhost:11434"
indicrag ask "Which grades have the bagless period?" --method hybrid --ollama
```

Windows PowerShell:

```powershell
$env:OLLAMA_MODEL = "your-installed-model-name"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
indicrag ask "Which grades have the bagless period?" --method hybrid --ollama
```

The generator receives the retrieved passages as evidence. Check its response against the citations; local generation can still produce unsupported text. `.env.example` documents the optional variables, but the application reads them from the process environment and does not automatically load a `.env` file.

## Project structure

```text
app.py                         Streamlit web app
src/indicrag/                  Ingestion, retrieval, answering, evaluation, and CLI
data/sample/passages.jsonl     Bundled English/Hindi starter passages
data/eval/qa.jsonl             Curated development/test questions
data/sources.json              Source document provenance
notebooks/                     Jupyter research walkthrough
docs/                          Architecture, evaluation, data card, and error analysis
legacy/                        Earlier NLP lab files, kept for reference
```

## Research background

The supplied papers inform the project design, but this repository does not reimplement each paper:

- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762): introduces the Transformer architecture. This project uses pretrained encoders rather than training a Transformer from scratch.
- Conneau et al. (2020), [Unsupervised Cross-lingual Representation Learning at Scale](https://aclanthology.org/2020.acl-main.747/): motivates multilingual pretraining and language-specific evaluation. XLM-R is not the project's answer generator.
- Feng et al. (2022), [Language-agnostic BERT Sentence Embedding](https://aclanthology.org/2022.acl-long.62/): motivates LaBSE for cross-lingual sentence retrieval.
- Rahimi et al. (2019), [Massively Multilingual Transfer for NER](https://aclanthology.org/P19-1015/): motivates checking transfer quality separately by language.

## Troubleshooting

- **`indicrag: command not found`**: activate `.venv` and install dependencies with `python -m pip install -r requirements.txt`.
- **`ModuleNotFoundError: indicrag`**: run commands from the repository root and confirm the active Python is the `.venv` interpreter.
- **LaBSE download fails**: use `--method bm25` or `--method tfidf` while offline; dense and hybrid retrieval need the model weights on first use.
- **Ollama is not reachable**: start the Ollama service, check that `OLLAMA_BASE_URL` points to it, and confirm the selected model is installed.
- **No answer or an unexpected abstention**: inspect the retrieved passages, retrieval method, and threshold. A low score means the current corpus may not contain useful evidence.

## More project notes

- [Architecture](docs/architecture.md)
- [Data card and provenance](docs/data-card.md)
- [Evaluation protocol](docs/evaluation.md)
- [Benchmark results and limits](docs/benchmark-results.md)
- [Difficult-case error analysis](docs/error-analysis.md)

The original policy source URLs are recorded in [`data/sources.json`](data/sources.json). For research-grade conclusions, ingest and cite the full source documents, expand the question set, and have the evidence and answer annotations reviewed by people fluent in the target languages and familiar with the policy domain.
