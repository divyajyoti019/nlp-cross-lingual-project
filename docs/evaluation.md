# Evaluation protocol

## Split discipline

Use dev to select answerability thresholds and prompts. Use test once for the final comparison. Add questions and source passages with stable IDs; keep the answer and evidence annotation hidden from model inputs.

The bundled set is a short 45-row software demonstration set (12 dev, 33 test). It is not the 300-500-example research-scale set recommended by the assignment. Expand it with additional public documents and human-reviewed questions before presenting results as a benchmark.

## Retrieval

Report Recall@K (fraction of annotated relevant passages retrieved), Precision@K (relevant retrieved passages divided by K), and MRR against gold_passage_ids for answerable questions. Compare BM25, character TF-IDF, LaBSE dense, and hybrid retrieval with the same corpus, questions, and K.

## Answer generation

Report exact match and Unicode-aware token F1 against gold_answer. Compare the deterministic extractive answerer with optional Ollama generation. If --compare-direct is used, report direct-model quality separately from RAG quality; do not mix them into retrieval metrics.

## Answerability

Report accuracy, precision, recall, F1, and the confusion matrix. Calibrate a threshold on dev, record that threshold, and apply it unchanged to test.

## Language and hard cases

Report results separately for English, Hindi, and code-mixed questions. Review all rows marked hard_case, especially numbers, dates, grade ranges, Romanized Hindi, similar passages, and unanswerable questions that share vocabulary with the policy.

## Error analysis

For at least 15 difficult examples in the final dataset, record:

| Field | What to inspect |
| --- | --- |
| Query type | English, Hindi, or code-mixed; note script and spelling |
| Gold passage | Does the source collection contain the answer? |
| Retrieval error | Missed passage, wrong language, or close distractor |
| Generation error | Unsupported detail, wrong language, or incomplete answer |
| Abstention error | False answer or false refusal |
| Proposed fix | Data, chunking, retrieval, threshold, prompt, or annotation |
