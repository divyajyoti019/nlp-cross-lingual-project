# NLP Lab 9 Solution

This solution implements the four tasks from the supplied lab: cross-lingual sentiment transfer, an LSTM encoder-decoder, multilingual NER, and multilingual clustering. Open `NLP_Lab_9_Solution.ipynb` in Jupyter and run its cells from top to bottom, or run the equivalent `NLP_Lab_9_Solution.py` script.

## Run it

Use Python 3.11 or later. In VS Code, install the Microsoft **Python**, **Pylance**, and **Jupyter** extensions, then open this project folder. The workspace is configured to use its `.venv` environment.

In the VS Code terminal, create the environment, install the project packages, and register its Jupyter kernel:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --prefix .venv --name nlp-project --display-name "Python (NLP project)"
python NLP_Lab_9_Solution.py
```

For notebooks, select the **Python (NLP project)** kernel in VS Code. To use JupyterLab outside VS Code, activate `.venv` and run `jupyter lab`.

The `.gitignore` excludes the virtual environment, downloaded datasets, model checkpoints, and generated outputs. Keep data and model files out of Git unless their licenses allow redistribution.

On first run, the script downloads the LaBSE sentence encoder, English GLUE/SST-2 data, the OPUS-100 English-Hindi parallel corpus, and the multilingual NER checkpoint. An internet connection is required for these downloads. The script writes CSV tables and PNG figures to `nlp_lab9_outputs/`.

For the LSTM training experiment, defaults are 3,000 parallel training pairs and 5 epochs. You can adjust them, for example:

```bash
python NLP_Lab_9_Solution.py --q2-train-limit 10000 --q2-epochs 15
```

## What each question reports

- **Q1:** LaBSE cosine similarities for English-Hindi parallel sentence pairs, English SST-2 validation accuracy, and accuracy plus a classification report for a small hand-labelled Hindi sentiment demonstration set.
- **Q2:** Token-weighted training loss by epoch, sample English-Hindi translations, and SacreBLEU on held-out OPUS-100 examples. The LSTM is trained from scratch; its final encoder hidden and cell states initialize the decoder.
- **Q3:** Predicted PER/LOC/ORG entities and confidence scores, plus mBERT WordPiece tokenization counts for English and Hindi examples.
- **Q4:** K-Means clusters, cosine silhouette, adjusted Rand index (ARI) against known topic labels, normalized mutual information (NMI) against language labels, contingency tables, and a PCA plot.

## Reporting limitations

The Hindi sentiment examples are a small curated demonstration set, and the clustering corpus contains manually written parallel examples. Their scores demonstrate the pipeline and should not be presented as broad benchmarks. The NER examples are illustrative and do not have a gold-labelled test set, so the script does not report NER F1. The LSTM is intentionally small enough for a lab run; translation quality depends on the training limit, epochs, hardware, and corpus sample. Report the values produced by your run and describe these limits.

The NER checkpoint is based on multilingual BERT and its model card lists training languages Arabic, German, English, Spanish, French, Italian, Latvian, Dutch, Portuguese, and Chinese. Since Hindi is absent from that list, the Hindi example probes transfer to an unseen language. OPUS-100 supplies the English-Hindi parallel data used for Q2. See the [NER model card](https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl) and [OPUS-100 dataset card](https://huggingface.co/datasets/Helsinki-NLP/opus-100).
