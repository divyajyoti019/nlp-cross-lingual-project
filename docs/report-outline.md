# Research report outline

1. **Abstract:** domain, languages, system variants, dataset size, main results.
2. **Problem and research questions:** cross-lingual retrieval, hybrid gains, grounded answers, abstention.
3. **Related work:** Transformer, XLM-R, LaBSE, multilingual transfer for low-resource settings.
4. **Data:** official source documents, collection date, languages, passage segmentation, QA annotation, split policy, licensing.
5. **Methods:** BM25, character TF-IDF, LaBSE dense, hybrid alpha, answerability calibration, extractive and optional LLM answerers.
6. **Experimental setup:** software versions, model IDs, hardware, K, thresholds, random seed, and latency.
7. **Results:** retrieval metrics, answer metrics, answerability metrics, per-language table, direct-vs-RAG ablation.
8. **Error analysis:** at least 15 difficult cases, with retrieval/generation/abstention error types.
9. **Limitations and ethics:** translation quality, language coverage, source freshness, hallucination, model license, privacy.
10. **Conclusion and next work.**
