"""Streamlit interface for the IndicRAG-QA demonstration."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from indicrag.generation import OllamaChatGenerator
from indicrag.ingest import load_passages
from indicrag.pipeline import IndicRAG
from indicrag.retrieval import DEFAULT_EMBEDDING_MODEL

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "sample" / "passages.jsonl"
EXTRA_PATH = ROOT / "data" / "processed" / "passages.jsonl"

st.set_page_config(page_title="IndicRAG-QA", page_icon="📚", layout="wide")
st.title("IndicRAG-QA")
st.caption("Evidence-grounded question answering for English, Hindi, and code-mixed queries")


@st.cache_resource(show_spinner="Preparing the passage index…")
def build_engine(embedding_model: str, alpha: float) -> IndicRAG:
    passages = load_passages(SAMPLE_PATH)
    if EXTRA_PATH.exists():
        passages.extend(load_passages(EXTRA_PATH))
    return IndicRAG(passages, embedding_model=embedding_model, alpha=alpha)


with st.sidebar:
    st.header("Retrieval settings")
    method = st.selectbox("Method", ["hybrid", "bm25", "tfidf", "dense"])
    top_k = st.slider("Passages to retrieve", min_value=1, max_value=10, value=5)
    threshold = st.slider("Answerability threshold", 0.0, 1.0, 0.42, 0.01)
    alpha = st.slider("Hybrid lexical weight", 0.0, 1.0, 0.55, 0.05)
    model_name = st.text_input("Multilingual embedding model", DEFAULT_EMBEDDING_MODEL)
    use_ollama = st.checkbox("Use a local Ollama generator")
    if use_ollama:
        st.caption("Set OLLAMA_MODEL and start Ollama before submitting a question.")
    st.divider()
    st.caption("First dense search downloads the selected embedding model.")
    st.caption("Starter evidence comes from the NEP 2020 English and Hindi policy sources.")

question = st.text_area(
    "Ask an education-policy question",
    placeholder="Example: NEP ke according coding activities kaunse school stage me start hoti hain?",
    height=90,
)

if st.button("Search and answer", type="primary", disabled=not question.strip()):
    try:
        engine = build_engine(model_name.strip(), alpha)
        if use_ollama:
            engine.ollama = OllamaChatGenerator(os.getenv("OLLAMA_MODEL"))
        with st.spinner("Retrieving evidence and preparing the answer…"):
            result = engine.ask(
                question,
                method=method,
                top_k=top_k,
                threshold=threshold,
                use_ollama=use_ollama,
            )
        st.subheader("Answer")
        if result["answerable"]:
            st.success(result["answer"])
        else:
            st.warning(result["answer"])
        st.caption(
            f"Detected query type: {result['query_language']} · "
            f"confidence: {result['confidence']:.3f} · "
            f"method: {result['retrieval_method']}"
        )
        st.subheader("Retrieved evidence")
        for index, citation in enumerate(result["citations"], start=1):
            heading = citation["title"]
            if citation["section"]:
                heading += f" · section {citation['section']}"
            with st.expander(f"{index}. {heading} · score {citation['score']:.3f}", expanded=index == 1):
                st.write(citation["excerpt"])
                st.caption(
                    f"Passage: {citation['passage_id']} · "
                    f"document: {citation['doc_id']} · language: {citation['language']}"
                )
                if citation["source_url"]:
                    st.markdown(f"[Open source document]({citation['source_url']})")
    except Exception as exc:
        st.error(str(exc))

st.divider()
st.caption(
    "Research demo: confidence thresholds require calibration on a development set. "
    "Always check cited source passages for important decisions."
)
