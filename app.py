"""Streamlit website for the IndicRAG-QA demonstration."""

from __future__ import annotations

import os
from pathlib import Path
import re

import streamlit as st

from indicrag.generation import OllamaChatGenerator
from indicrag.ingest import ingest_file, load_passages, write_passages
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
        --ink: #172d35;
        --muted: #667a7d;
        --teal: #087f78;
        --teal-dark: #075b58;
        --line: #dce7e3;
      }
      [data-testid="stAppViewContainer"] {
        background: #f6f8f6;
      }
      [data-testid="stMainBlockContainer"] {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
      }
      [data-testid="stSidebar"] {
        background: #eef3f0;
        border-right: 1px solid var(--line);
      }
      .hero {
        position: relative;
        overflow: hidden;
        padding: 2.6rem 2.8rem;
        border: 1px solid #d6e5dc;
        border-radius: 26px;
        background: linear-gradient(115deg, #e5f0e9 0%, #f2f2e8 62%, #f5ece0 100%);
        box-shadow: 0 18px 50px rgba(23, 45, 53, .07);
        margin: .3rem 0 1.5rem 0;
      }
      .hero:after { content: "अ"; position: absolute; right: 5%; top: -3.6rem; color: rgba(8,127,120,.07); font-size: 18rem; font-weight: 800; line-height: 1; pointer-events: none; }
      .hero-kicker, .section-kicker {
        color: var(--teal-dark);
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: .55rem;
      }
      .hero h1 {
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 3.45rem);
        line-height: 1.08;
        letter-spacing: -.035em;
        margin: 0 0 .8rem 0;
        position: relative; z-index: 1;
      }
      .hero p {
        color: #425d68;
        font-size: 1.05rem;
        line-height: 1.6;
        max-width: 720px;
        margin: 0;
        position: relative; z-index: 1;
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
      .trust-strip { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.25rem; position: relative; z-index: 1; }
      .trust-strip span { border: 1px solid rgba(8,91,88,.16); border-radius: 999px; padding: .35rem .7rem; background: rgba(255,255,255,.55); color: #345b58; font-size: .78rem; font-weight: 650; }
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
        background: #f2f6f4;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: .8rem 1rem;
      }
      div[data-testid="stTabs"] button { font-weight: 650; }
      div[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 1.2rem; }
      div[data-testid="stForm"] { background: #fff; padding: 1rem 1.1rem .8rem; border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 8px 24px rgba(23,45,53,.035); }
      .team-card {
        height: 100%;
        min-height: 230px;
        box-sizing: border-box;
        padding: 1.15rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: linear-gradient(160deg, #ffffff 0%, #f5f8f7 100%);
      }
      .team-number {
        display: inline-grid;
        place-items: center;
        width: 2.5rem;
        height: 2.5rem;
        margin-bottom: .9rem;
        border-radius: 13px;
        color: var(--teal-dark);
        background: #e3f1ee;
        font-size: .88rem;
        font-weight: 800;
      }
      .team-name {
        color: var(--ink);
        font-size: 1.02rem;
        font-weight: 750;
        margin-bottom: .3rem;
      }
      .team-role {
        color: var(--teal-dark);
        font-size: .9rem;
        font-weight: 700;
        line-height: 1.35;
        margin-bottom: .65rem;
      }
      .team-description {
        color: var(--muted);
        font-size: .88rem;
        line-height: 1.5;
      }
      @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
          padding-top: 1rem;
          padding-left: 1rem;
          padding-right: 1rem;
        }
        .hero { padding: 1.55rem; border-radius: 18px; }
        .hero:after { right: -1.5rem; top: 0; font-size: 10rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_corpus(include_extended: bool = False):
    passages = load_passages(SAMPLE_PATH)
    if include_extended and EXTRA_PATH.exists():
        passages.extend(load_passages(EXTRA_PATH))
    return passages


def corpus_stamp(include_extended: bool) -> str:
    if not include_extended or not EXTRA_PATH.exists():
        return "starter"
    stat = EXTRA_PATH.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


@st.cache_resource(show_spinner="Preparing the passage index…")
def build_engine(
    embedding_model: str, alpha: float, include_extended: bool, data_version: str
) -> IndicRAG:
    passages = load_corpus(include_extended)
    return IndicRAG(passages, embedding_model=embedding_model, alpha=alpha)


def _choose_example(question: str) -> None:
    st.session_state["question_input"] = question


with st.sidebar:
    st.markdown(
        '<div class="project-mark"><span>IR</span>IndicRAG-QA</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Retrieval controls")
    st.caption("Tune how policy passages are found. Settings apply to your next search.")
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
    include_extended = st.toggle(
        "Use expanded corpus",
        value=EXTRA_PATH.exists(),
        help="Includes the official full-text documents or any passages you add in the Corpus tab.",
    )
    corpus_total = len(load_corpus(include_extended))
    st.caption(f"Searching **{corpus_total:,} passages**")
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
      <div class="hero-kicker">IndicRAG-QA &nbsp;·&nbsp; Research workspace</div>
      <h1>Explore policy.<br>Follow the evidence.</h1>
      <p>Ask about India’s National Education Policy 2020 in English, Hindi, or Hinglish. Get a concise response grounded in passages you can inspect.</p>
      <div class="trust-strip"><span>English · हिन्दी · Hinglish</span><span>Evidence-linked answers</span><span>Abstains when evidence is weak</span></div>
    </section>
    """,
    unsafe_allow_html=True,
)

ask_tab, compare_tab, corpus_tab, about_tab = st.tabs(
    ["💬 Ask", "⚖️ Compare methods", "🗂️ Explore corpus", "📘 About"]
)

with ask_tab:
    st.markdown('<div class="section-kicker">Start with a question</div>', unsafe_allow_html=True)
    st.caption("Choose a prompt to get started, or write a question in your own words.")
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
            "Find an answer and its sources  →",
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
                engine = build_engine(
                    model_name.strip(), alpha, include_extended, corpus_stamp(include_extended)
                )
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

        st.markdown("### Evidence used")
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
        '<p class="small-note">Research demo: check the linked source passages before relying on an answer. '
        "The starter collection contains curated NEP summaries; the expanded collection adds official full texts.</p>",
        unsafe_allow_html=True,
    )

with compare_tab:
    st.markdown('<div class="section-kicker">Experiment with retrieval</div>', unsafe_allow_html=True)
    st.markdown("### Same question, different ranking")
    compare_question = st.text_input(
        "Question to compare",
        value=st.session_state.get("question_input", ""),
        key="compare_question",
        placeholder="Ask about a policy topic…",
    )
    method_options = {
        "BM25 · keywords": "bm25",
        "Character TF-IDF": "tfidf",
        "LaBSE · multilingual": "dense",
        "Hybrid": "hybrid",
    }
    compare_labels = st.multiselect(
        "Methods",
        list(method_options),
        default=["BM25 · keywords", "Character TF-IDF", "Hybrid"],
    )
    if st.button("Compare rankings", type="primary", disabled=not compare_question.strip() or not compare_labels):
        try:
            engine = build_engine(
                model_name.strip(), alpha, include_extended, corpus_stamp(include_extended)
            )
            compared = {}
            with st.spinner("Running each retriever over the selected corpus…"):
                for label in compare_labels:
                    compared[label] = engine.retrieve(
                        compare_question, method=method_options[label], top_k=top_k
                    )
            st.session_state["comparison_results"] = compared
            st.session_state["comparison_query"] = compare_question
        except Exception as exc:
            st.error(str(exc))

    compared = st.session_state.get("comparison_results")
    if compared:
        st.caption(f"Question: {st.session_state.get('comparison_query', '')}")
        summary_rows = []
        score_chart = []
        for label, hits in compared.items():
            if hits:
                score_chart.append({"Method": label, "Top score": hits[0].score})
                summary_rows.append({
                    "Method": label,
                    "Top score": round(hits[0].score, 3),
                    "Top source": hits[0].passage.title,
                    "Section": hits[0].passage.section,
                })
        if summary_rows:
            st.dataframe(summary_rows, hide_index=True, use_container_width=True)
            with st.expander("Compare top-match scores"):
                st.bar_chart(score_chart, x="Method", y="Top score", horizontal=True)
        for label, hits in compared.items():
            with st.expander(f"{label} · {len(hits)} passages"):
                for hit in hits:
                    p = hit.passage
                    st.markdown(f"**{hit.rank}. {p.title}** · score `{hit.score:.3f}`")
                    if p.section:
                        st.caption(p.section)
                    st.write(p.text)
                    st.divider()
    st.info("BM25 and TF-IDF run locally without model downloads. Dense and Hybrid use LaBSE and may download its weights on first use.")

with corpus_tab:
    st.markdown('<div class="section-kicker">Inspect and grow the collection</div>', unsafe_allow_html=True)
    st.markdown("### Education policy library")
    st.caption("Filter the passages that the selected corpus makes searchable, or add a policy document of your own.")
    if message := st.session_state.pop("corpus_notice", None):
        st.success(message)

    corpus_passages = load_corpus(include_extended)
    doc_titles = sorted({p.title for p in corpus_passages})
    doc_languages = sorted({p.language for p in corpus_passages})
    m1, m2, m3 = st.columns(3)
    m1.metric("Passages", f"{len(corpus_passages):,}")
    m2.metric("Documents", len(doc_titles))
    m3.metric("Languages", len(doc_languages))

    filter_columns = st.columns([1, 1, 2])
    chosen_language = filter_columns[0].selectbox("Language", ["All", *doc_languages])
    chosen_title = filter_columns[1].selectbox("Document", ["All", *doc_titles])
    search_text = filter_columns[2].text_input("Find words in passages", placeholder="e.g. foundational literacy")
    filtered = [
        p for p in corpus_passages
        if (chosen_language == "All" or p.language == chosen_language)
        and (chosen_title == "All" or p.title == chosen_title)
        and (not search_text.strip() or search_text.casefold() in p.text.casefold())
    ]
    st.caption(f"Showing {len(filtered):,} matching passages")
    page_size = 10
    page_count = max(1, (len(filtered) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1)
    start = (int(page) - 1) * page_size
    for p in filtered[start : start + page_size]:
        heading = f"{p.title} · {p.language}"
        if p.section:
            heading += f" · {p.section}"
        with st.expander(heading):
            st.write(p.text)
            st.caption(f"Passage `{p.passage_id}`" + (f" · Page {p.page}" if p.page else ""))
            if p.source_url:
                st.link_button("Open source", p.source_url)

    st.markdown("#### Add a source document")
    st.caption("Upload a text-based PDF, TXT, or Markdown file. It will be segmented into searchable passages locally.")
    upload = st.file_uploader("Choose a policy document", type=["pdf", "txt", "md"], key="corpus_upload")
    if upload and st.button("Index document", type="primary"):
        try:
            safe_name = Path(upload.name).name
            raw_dir = ROOT / "data" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            source_path = raw_dir / safe_name
            source_path.write_bytes(upload.getvalue())
            doc_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(safe_name).stem).strip("-").lower()
            new_passages = ingest_file(source_path, doc_id=doc_id, title=Path(safe_name).stem, language="auto")
            existing = load_passages(EXTRA_PATH) if EXTRA_PATH.exists() else []
            existing = [p for p in existing if p.doc_id != doc_id]
            write_passages([*existing, *new_passages], EXTRA_PATH)
            st.cache_resource.clear()
            st.session_state["corpus_notice"] = (
                f"Indexed {len(new_passages):,} passages from {safe_name}. "
                "Enable the expanded corpus in the sidebar to search it."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Could not index document: {exc}")

with about_tab:
    st.markdown('<div class="section-kicker">About this project</div>', unsafe_allow_html=True)
    st.markdown("### Cross-lingual education research, with inspectable evidence")
    st.write(
        "IndicRAG-QA explores whether English, Hindi, and code-mixed questions can retrieve "
        "useful evidence from a bilingual education-policy corpus. It is designed for comparison "
        "and error analysis, not as a production policy-advice service."
    )
    sample_count = len(load_passages(SAMPLE_PATH))
    expanded_count = len(load_passages(EXTRA_PATH)) if EXTRA_PATH.exists() else 0
    metric_columns = st.columns(3)
    metric_columns[0].metric("Starter passages", f"{sample_count:,}")
    metric_columns[1].metric("Expanded passages", f"{expanded_count:,}")
    metric_columns[2].metric("Curated evaluation questions", "45")
    st.markdown("### Expanded education sources")
    st.write(
        "Run `python scripts/build_extended_corpus.py` to download and index the complete NEP 2020 "
        "documents in English and Hindi plus the UDISE+ 2024–25 and 2025–26 NEP-structure reports. "
        "This adds recent school statistics to the policy text. The source PDFs are kept in the ignored "
        "`data/raw/` folder; generated passages are kept in `data/processed/`."
    )
    st.markdown("### Another large multilingual QA dataset")
    st.write(
        "AI4Bharat’s Indic-RAG-Suite lists 21.4 million samples across 18 Indian languages and English, "
        "with an 8.82 GB download. It is based on Wikipedia and generated QA pairs, so it is useful for "
        "multilingual retrieval experiments but has a different scope from official education policy."
    )
    st.link_button("Open Indic-RAG-Suite dataset", "https://huggingface.co/datasets/ai4bharat/Indic-Rag-Suite")
    st.markdown("### Evaluation notes")
    st.write(
        "The 45 evaluation questions remain a small curated set (12 development and 33 test examples). "
        "Use them to reproduce the software baseline and inspect errors; do not treat them as broad "
        "real-world performance evidence."
    )
    st.markdown("### Included in the repository")
    st.markdown(
        "- Streamlit question-answering interface and command-line tools\n"
        "- PDF, text, and Markdown passage ingestion\n"
        "- BM25, character TF-IDF, LaBSE, and hybrid retrieval\n"
        "- Evaluation protocol, benchmark snapshot, data card, and research notebook"
    )
    st.caption("The expanded corpus follows official source URLs documented in `data/sources.json`.")
    st.markdown("### Meet the team")
    st.write("Suggested responsibilities for a four-member project team:")
    team_members = [
        (
            "01",
            "Member 1",
            "Project Lead & Research Coordinator",
            "Sets the project goals, coordinates the research work, and brings the final report and presentation together.",
        ),
        (
            "02",
            "Member 2",
            "NLP & Retrieval Engineer",
            "Builds and compares BM25, character TF-IDF, LaBSE, and hybrid passage retrieval.",
        ),
        (
            "03",
            "Member 3",
            "Data & Evaluation Lead",
            "Maintains passages and QA annotations, runs evaluation, and reviews language-specific errors.",
        ),
        (
            "04",
            "Member 4",
            "Web App & Documentation Lead",
            "Improves the Streamlit experience and keeps setup instructions and the project demo reproducible.",
        ),
    ]
    team_columns = st.columns(4)
    for column, (number, name, role, description) in zip(team_columns, team_members):
        with column:
            st.markdown(
                f'<div class="team-card">'
                f'<div class="team-number">{number}</div>'
                f'<div class="team-name">{name}</div>'
                f'<div class="team-role">{role}</div>'
                f'<div class="team-description">{description}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
    st.caption("Replace Member 1–4 with your teammates’ names and adjust responsibilities to match their contributions.")

st.markdown("---")
st.caption("IndicRAG-QA · Evidence-grounded multilingual NLP research demo")
