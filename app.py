"""Streamlit website for the IndicRAG-QA demonstration."""

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

st.set_page_config(
    page_title="IndicRAG-QA | Policy question answering",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
      :root {
        --ink: #142b3a;
        --muted: #5f7480;
        --teal: #087f83;
        --teal-dark: #075f66;
        --sand: #f4f0e8;
        --line: #dce5e4;
      }
      [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f7faf9 0%, #ffffff 340px);
      }
      [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
      }
      [data-testid="stSidebar"] {
        background: #f2f6f5;
        border-right: 1px solid var(--line);
      }
      .hero {
        padding: 2.1rem 2.25rem;
        border: 1px solid #d7e7e3;
        border-radius: 24px;
        background:
          radial-gradient(circle at 92% 15%, rgba(255,255,255,.72), transparent 27%),
          linear-gradient(120deg, #e6f3ef 0%, #f7f1e7 100%);
        box-shadow: 0 14px 42px rgba(20, 43, 58, .06);
        margin-bottom: 1.5rem;
      }
      .hero-kicker, .section-kicker {
        color: var(--teal-dark);
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .55rem;
      }
      .hero h1 {
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1.08;
        letter-spacing: -.035em;
        margin: 0 0 .7rem 0;
      }
      .hero p {
        color: #425d68;
        font-size: 1.05rem;
        line-height: 1.6;
        max-width: 780px;
        margin: 0;
      }
      .project-mark {
        color: var(--ink);
        font-weight: 800;
        font-size: 1.05rem;
        margin: .25rem 0 1.2rem 0;
      }
      .project-mark span {
        display: inline-grid;
        place-items: center;
        width: 2rem;
        height: 2rem;
        margin-right: .55rem;
        border-radius: 10px;
        color: white;
        background: var(--teal);
      }
      .small-note {
        color: var(--muted);
        font-size: .9rem;
        line-height: 1.5;
      }
      div.stButton > button[kind="primary"],
      div.stFormSubmitButton > button[kind="primary"] {
        background: var(--teal);
        border-color: var(--teal);
        border-radius: 12px;
        min-height: 2.8rem;
      }
      div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 11px;
      }
      div[data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 13px;
        background: rgba(255,255,255,.8);
      }
      div[data-testid="stMetric"] {
        background: #f5f8f7;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: .8rem 1rem;
      }
      @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
          padding-top: 1rem;
          padding-left: 1rem;
          padding-right: 1rem;
        }
        .hero { padding: 1.5rem; border-radius: 18px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing the passage index…")
def build_engine(embedding_model: str, alpha: float) -> IndicRAG:
    passages = load_passages(SAMPLE_PATH)
    if EXTRA_PATH.exists():
        passages.extend(load_passages(EXTRA_PATH))
    return IndicRAG(passages, embedding_model=embedding_model, alpha=alpha)


def _choose_example(question: str) -> None:
    st.session_state["question_input"] = question


with st.sidebar:
    st.markdown(
        '<div class="project-mark"><span>IR</span>IndicRAG-QA</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Search settings")
    method_label = st.selectbox(
        "Retrieval method",
        ["Hybrid · recommended", "BM25 · keyword", "Character TF-IDF", "LaBSE · multilingual"],
        help="Hybrid combines lexical matching with multilingual sentence similarity.",
    )
    method_by_label = {
        "Hybrid · recommended": "hybrid",
        "BM25 · keyword": "bm25",
        "Character TF-IDF": "tfidf",
        "LaBSE · multilingual": "dense",
    }
    method = method_by_label[method_label]
    top_k = st.slider("Passages to retrieve", min_value=1, max_value=10, value=5)
    threshold = st.slider(
        "Answerability threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.42,
        step=0.01,
        help="The top passage must score at or above this cutoff for the system to answer.",
    )
    alpha = st.slider(
        "Hybrid lexical weight",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05,
        help="Used only by Hybrid. Higher values give more weight to word overlap.",
    )
    model_name = st.text_input(
        "Embedding model",
        value=DEFAULT_EMBEDDING_MODEL,
        help="LaBSE is used by Dense and Hybrid retrieval.",
    )
    use_ollama = st.toggle(
        "Use a local Ollama answer model",
        value=False,
        help="Optional. The default answerer selects a sentence directly from retrieved passages.",
    )
    if use_ollama:
        st.caption("Set `OLLAMA_MODEL` in the environment and start Ollama before searching.")
    st.divider()
    st.markdown(
        '<div class="small-note"><b>First dense search</b><br>'
        "LaBSE model weights may need to download once. Choose BM25 or TF-IDF for a quick "
        "first try without that download.</div>",
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">A multilingual policy research demo</div>
      <h1>Find policy answers<br>with evidence you can inspect.</h1>
      <p>Ask about India’s National Education Policy 2020 in English, Hindi, or Hinglish.<br>
      IndicRAG-QA retrieves relevant passages, gives a concise answer, and shows the source text.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

ask_tab, method_tab, about_tab = st.tabs(["💬 Ask a question", "🧭 How it works", "📘 Project notes"])

with ask_tab:
    st.markdown('<div class="section-kicker">Try a sample question</div>', unsafe_allow_html=True)
    examples = [
        ("English", "At what school stage are coding activities introduced?"),
        ("हिन्दी", "कोडिंग गतिविधियाँ किस चरण में शुरू की जाती हैं?"),
        ("Hinglish", "NEP ke according coding activities kis stage se shuru hoti hain?"),
    ]
    example_columns = st.columns(3)
    for column, (language, example) in zip(example_columns, examples):
        with column:
            st.button(
                f"Use {language} example",
                key=f"example_{language}",
                use_container_width=True,
                on_click=_choose_example,
                args=(example,),
            )

    with st.form("question_form", clear_on_submit=False):
        question = st.text_area(
            "Your question",
            key="question_input",
            placeholder="Ask about curriculum, school stages, vocational education, or another NEP topic…",
            height=115,
            help="Questions can be written in English, Hindi, or Hindi-English code-mixed text.",
        )
        submitted = st.form_submit_button(
            "Search policy evidence",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not question.strip():
            st.session_state["last_result"] = None
            st.session_state["last_error"] = "Enter a question or choose one of the examples above."
        else:
            st.session_state["last_result"] = None
            st.session_state["last_error"] = None
            try:
                engine = build_engine(model_name.strip(), alpha)
                if use_ollama:
                    engine.ollama = OllamaChatGenerator(os.getenv("OLLAMA_MODEL"))
                with st.spinner("Searching the policy passages…"):
                    st.session_state["last_result"] = engine.ask(
                        question,
                        method=method,
                        top_k=top_k,
                        threshold=threshold,
                        use_ollama=use_ollama,
                    )
            except Exception as exc:
                st.session_state["last_error"] = str(exc)

    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])

    result = st.session_state.get("last_result")
    if result:
        st.markdown("---")
        answer_column, status_column = st.columns([4, 1])
        with answer_column:
            st.markdown("### Answer")
            if result["answerable"]:
                st.success(result["answer"])
            else:
                st.warning(result["answer"])
        with status_column:
            st.metric("Top match", f"{result['confidence']:.3f}")
            st.caption(f"Query type: **{result['query_language']}**")

        st.markdown("### Retrieved evidence")
        st.caption(
            f"Method: {result['retrieval_method']} · "
            f"Answerability cutoff: {result['threshold']:.2f}"
        )
        if result["citations"]:
            for index, citation in enumerate(result["citations"], start=1):
                heading = citation["title"]
                if citation["section"]:
                    heading += f" · {citation['section']}"
                with st.expander(
                    f"{index}. {heading}  ·  score {citation['score']:.3f}",
                    expanded=index == 1,
                ):
                    st.write(citation["excerpt"])
                    metadata = [
                        f"Passage `{citation['passage_id']}`",
                        f"Language: {citation['language']}",
                    ]
                    if citation["page"]:
                        metadata.append(f"Page: {citation['page']}")
                    st.caption(" · ".join(metadata))
                    if citation["source_url"]:
                        st.link_button("Open source document", citation["source_url"])
        else:
            st.info("No passages were retrieved for this question.")

    st.markdown(
        '<p class="small-note">This is a research demo. Check source passages before relying on an answer; '
        "the bundled corpus contains curated NEP summaries, not the full policy document.</p>",
        unsafe_allow_html=True,
    )

with method_tab:
    st.markdown('<div class="section-kicker">The retrieval options</div>', unsafe_allow_html=True)
    st.markdown("### One question, four ways to find evidence")
    method_columns = st.columns(4)
    method_cards = [
        ("BM25", "Keyword matching", "A fast baseline for questions that use the same words as the policy passage."),
        ("Character TF-IDF", "Spelling-aware matching", "Compares short character patterns and can help with spelling and word-form differences."),
        ("LaBSE", "Cross-language meaning", "Uses multilingual sentence embeddings to connect related English and Hindi passages."),
        ("Hybrid", "Combined retrieval", "Blends lexical evidence with LaBSE similarity; adjust the weight in the sidebar."),
    ]
    for column, (name, short, detail) in zip(method_columns, method_cards):
        with column:
            st.markdown(f"#### {name}")
            st.caption(short)
            st.write(detail)

    st.markdown("### Answering flow")
    st.markdown(
        "**Question → passage ranking → answerability check → answer and source passages.** "
        "The default answerer selects a sentence from retrieved text. A local Ollama model is optional; "
        "it receives the retrieved passages as evidence."
    )
    st.info("Dense and Hybrid retrieval download LaBSE the first time they are used. BM25 and TF-IDF work without model weights.")

with about_tab:
    st.markdown('<div class="section-kicker">About this project</div>', unsafe_allow_html=True)
    st.markdown("### A small, inspectable cross-lingual QA baseline")
    st.write(
        "IndicRAG-QA explores whether English, Hindi, and code-mixed questions can retrieve "
        "useful evidence from a bilingual education-policy corpus. It is designed for comparison "
        "and error analysis, not as a production policy-advice service."
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Starter passages", "30")
    metric_columns[1].metric("Curated questions", "45")
    metric_columns[2].metric("Question languages", "3")
    st.markdown("### What the results mean")
    st.write(
        "The bundled data is intentionally small: 12 development questions and 33 test questions. "
        "Use it to reproduce the software baseline and inspect errors. Expand and independently "
        "review the data before making broad performance claims."
    )
    st.markdown("### Included in the repository")
    st.markdown(
        "- Streamlit question-answering interface and command-line tools\n"
        "- PDF, text, and Markdown passage ingestion\n"
        "- BM25, character TF-IDF, LaBSE, and hybrid retrieval\n"
        "- Evaluation protocol, benchmark snapshot, data card, and research notebook"
    )
    st.caption("Primary starter sources: the English and Hindi editions of India’s National Education Policy 2020.")

st.markdown("---")
st.caption("IndicRAG-QA · Evidence-grounded multilingual NLP research demo")
