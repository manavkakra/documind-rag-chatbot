from __future__ import annotations

import html
import logging
from typing import List

import streamlit as st

from src.config import AppConfig
from src.embeddings import EmbeddingModel
from src.flashcard_generator import FlashcardGenerator
from src.pdf_loader import PDFLoader
from src.rag_pipeline import RAGPipeline
from src.retriever import HybridRetriever, RetrievedChunk
from src.summarizer import DocumentSummarizer
from src.text_splitter import DocumentChunker
from src.utils import confidence_label, format_citation, save_uploaded_file, setup_logging
from src.vector_database import FAISSVectorDatabase


setup_logging()
LOGGER = logging.getLogger(__name__)


def initialize_session_state() -> None:
    defaults = {
        "messages": [],
        "page_documents": [],
        "chunks": [],
        "vector_db": None,
        "retriever": None,
        "rag_pipeline": None,
        "summary": "",
        "flashcards": "",
        "knowledge_base_ready": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str) -> EmbeddingModel:
    return EmbeddingModel(model_name=model_name)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --dm-bg: #000000;
            --dm-bg-soft: #000000;
            --dm-surface: #1c1c1e;
            --dm-surface-2: #2c2c2e;
            --dm-ink: #f2f2f7;
            --dm-muted: #8e8e93;
            --dm-border: #38383a;
            --dm-blue: #0a84ff;
            --dm-blue-dark: #64d2ff;
            --dm-green: #30d158;
            --dm-amber: #ffd60a;
            --dm-red: #ff8a80;
        }

        .stApp {
            background: var(--dm-bg);
            color: var(--dm-ink);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.4rem;
            max-width: 1180px;
        }

        [data-testid="stSidebar"] {
            background: #000000;
            border-right: 1px solid var(--dm-border);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: var(--dm-ink);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--dm-muted);
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--dm-ink);
        }

        .dm-hero {
            background: var(--dm-surface);
            border: 1px solid var(--dm-border);
            border-radius: 22px;
            padding: 1.2rem 1.35rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.34);
        }

        .dm-hero-row {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1.2rem;
            flex-wrap: wrap;
        }

        .dm-brand {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            margin-bottom: 0.55rem;
        }

        .dm-logo {
            width: 42px;
            height: 42px;
            display: grid;
            place-items: center;
            border-radius: 14px;
            background: var(--dm-blue);
            color: #ffffff;
            border: 1px solid var(--dm-blue);
            font-weight: 800;
            font-size: 1.1rem;
        }

        .dm-title {
            margin: 0;
            font-size: 2rem;
            line-height: 1.08;
            font-weight: 760;
        }

        .dm-subtitle {
            margin: 0;
            color: var(--dm-muted);
            font-size: 0.98rem;
            max-width: 720px;
        }

        .dm-chip-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            align-items: center;
            justify-content: flex-end;
        }

        .dm-chip {
            display: inline-flex;
            align-items: center;
            border: 1px solid var(--dm-border);
            border-radius: 999px;
            padding: 0.34rem 0.64rem;
            background: var(--dm-surface-2);
            color: var(--dm-muted);
            font-size: 0.8rem;
            font-weight: 650;
            white-space: nowrap;
        }

        .dm-chip-ready {
            color: #ffffff;
            background: rgba(10, 132, 255, 0.28);
            border-color: var(--dm-blue);
        }

        .dm-chip-waiting {
            color: var(--dm-amber);
            background: rgba(255, 214, 10, 0.12);
            border-color: rgba(255, 214, 10, 0.42);
        }

        .dm-panel {
            background: var(--dm-surface);
            border: 1px solid var(--dm-border);
            border-radius: 18px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.28);
        }

        .dm-panel-title {
            margin: 0 0 0.25rem 0;
            font-size: 0.92rem;
            font-weight: 760;
            color: var(--dm-ink);
        }

        .dm-panel-copy {
            margin: 0;
            font-size: 0.86rem;
            color: var(--dm-muted);
            line-height: 1.45;
        }

        .dm-source-card {
            border: 1px solid var(--dm-border);
            border-radius: 18px;
            padding: 0.9rem;
            margin: 0.7rem 0;
            background: var(--dm-surface-2);
        }

        .dm-source-topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.7rem;
            margin-bottom: 0.45rem;
        }

        .dm-source-title {
            font-weight: 720;
            color: var(--dm-ink);
            font-size: 0.92rem;
        }

        .dm-score {
            font-size: 0.78rem;
            color: #ffffff;
            background: var(--dm-blue);
            border: 1px solid var(--dm-blue);
            border-radius: 999px;
            padding: 0.18rem 0.48rem;
            white-space: nowrap;
        }

        .dm-source-text {
            color: var(--dm-muted);
            font-size: 0.88rem;
            line-height: 1.48;
            margin: 0;
        }

        .dm-empty {
            border: 1px dashed var(--dm-border);
            border-radius: 18px;
            padding: 1.1rem;
            background: var(--dm-surface);
        }

        .dm-empty-title {
            margin: 0 0 0.35rem 0;
            font-weight: 760;
            color: var(--dm-ink);
        }

        .dm-empty-copy {
            color: var(--dm-muted);
            margin: 0;
            line-height: 1.5;
        }

        div[data-testid="stMetric"] {
            background: var(--dm-surface);
            border: 1px solid var(--dm-border);
            border-radius: 18px;
            padding: 0.75rem 0.85rem;
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--dm-ink);
        }

        .stButton > button {
            border-radius: 16px;
            border: 1px solid var(--dm-blue);
            font-weight: 700;
            background: var(--dm-blue);
            color: #ffffff;
        }

        .stButton > button[kind="primary"] {
            background: var(--dm-blue);
            border-color: var(--dm-blue);
            color: #ffffff;
        }

        .stButton > button:disabled,
        .stButton > button[kind="primary"]:disabled {
            background: rgba(10, 132, 255, 0.22);
            border-color: rgba(10, 132, 255, 0.36);
            color: rgba(242, 242, 247, 0.52);
            opacity: 1;
        }

        [data-testid="stSidebar"] .stButton > button {
            background: var(--dm-blue);
            border-color: var(--dm-blue);
            color: #ffffff !important;
            box-shadow: 0 10px 24px rgba(10, 132, 255, 0.26);
        }

        [data-testid="stSidebar"] .stButton > button *,
        [data-testid="stSidebar"] .stButton > button p,
        [data-testid="stSidebar"] .stButton > button span,
        [data-testid="stSidebar"] .stButton > button div {
            color: #ffffff !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: #409cff;
            border-color: #409cff;
            color: #ffffff;
        }

        [data-testid="stSidebar"] .stButton > button:disabled,
        [data-testid="stSidebar"] .stButton > button[kind="primary"]:disabled {
            background: rgba(10, 132, 255, 0.24);
            border-color: rgba(10, 132, 255, 0.38);
            color: #ffffff !important;
            box-shadow: none;
            opacity: 1;
        }

        [data-testid="stSidebar"] .stButton > button:disabled *,
        [data-testid="stSidebar"] .stButton > button[kind="primary"]:disabled * {
            color: #ffffff !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: var(--dm-surface);
            border: 1px dashed var(--dm-border);
            border-radius: 18px;
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: var(--dm-ink) !important;
        }

        [data-testid="stFileUploaderDropzone"] small {
            color: var(--dm-muted) !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background: var(--dm-blue);
            border: 1px solid var(--dm-blue);
            color: #ffffff;
            border-radius: 14px;
        }

        [data-testid="stFileUploaderDropzone"] button * {
            color: #ffffff !important;
        }

        [data-testid="stChatInput"] {
            background: var(--dm-surface);
            border: 1px solid var(--dm-border);
            border-radius: 24px;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.38);
        }

        [data-testid="stChatInput"] textarea {
            color: var(--dm-ink);
            background: var(--dm-surface);
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #78859a;
        }

        [data-testid="stChatInput"] button {
            color: var(--dm-blue);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 14px 14px 0 0;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }

        .stTabs [data-baseweb="tab"] p {
            color: #ffffff;
            font-weight: 700;
        }

        .stTabs [aria-selected="true"] p {
            color: #ffffff;
        }

        [data-testid="stChatMessage"] {
            border-radius: 24px;
            border: 1px solid var(--dm-border);
            background: var(--dm-surface);
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.32);
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: var(--dm-blue);
            border-color: var(--dm-blue);
            margin-left: auto;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
            color: #ffffff !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            background: var(--dm-surface-2);
            border-color: var(--dm-border);
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span {
            color: var(--dm-ink);
        }

        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p {
            color: var(--dm-muted);
        }

        [data-testid="stExpander"] {
            background: var(--dm-surface);
            border: 1px solid var(--dm-border);
            border-radius: 18px;
        }

        hr {
            border-color: var(--dm-border);
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .dm-title {
                font-size: 1.55rem;
            }

            .dm-chip-row {
                justify-content: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(config: AppConfig) -> None:
    status_class = "dm-chip-ready" if st.session_state.knowledge_base_ready else "dm-chip-waiting"
    status_text = "Knowledge base ready" if st.session_state.knowledge_base_ready else "Upload PDFs to begin"
    api_class = "dm-chip-ready" if config.gemini_api_key else "dm-chip-waiting"
    api_text = "Gemini configured" if config.gemini_api_key else "Gemini key missing"

    st.markdown(
        f"""
        <section class="dm-hero">
            <div class="dm-hero-row">
                <div>
                    <div class="dm-brand">
                        <div class="dm-logo">DM</div>
                        <div>
                            <h1 class="dm-title">DocuMind</h1>
                        </div>
                    </div>
                    <p class="dm-subtitle">
                        Ask grounded questions across multiple PDFs, inspect citations,
                        and turn documents into summaries and revision flashcards.
                    </p>
                </div>
                <div class="dm-chip-row">
                    <span class="dm-chip {status_class}">{status_text}</span>
                    <span class="dm-chip {api_class}">{api_text}</span>
                    <span class="dm-chip">Top {config.top_k} retrieval</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_intro() -> None:
    st.markdown(
        """
        <div class="dm-panel">
            <p class="dm-panel-title">Workflow</p>
            <p class="dm-panel-copy">
                Upload PDFs, build the knowledge base, then ask questions or generate
                study material from the indexed content.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_knowledge_base(uploaded_files: List[object], config: AppConfig) -> None:
    if not uploaded_files:
        st.warning("Upload at least one PDF before building the knowledge base.")
        return

    saved_paths = [save_uploaded_file(file, config.data_dir) for file in uploaded_files]
    loader = PDFLoader()
    chunker = DocumentChunker(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    embedding_model = get_embedding_model(config.embedding_model_name)

    page_documents = loader.load_many(saved_paths)
    chunks = chunker.split_documents(page_documents)

    if not chunks:
        st.error("No extractable text was found in the uploaded PDFs.")
        return

    vector_db = FAISSVectorDatabase(embedding_model=embedding_model)
    vector_db.create_index(chunks)
    vector_db.save_index(config.vector_store_dir)

    retriever = HybridRetriever(vector_db=vector_db, top_k=config.top_k)
    rag_pipeline = RAGPipeline(
        retriever=retriever,
        gemini_api_key=config.gemini_api_key,
        model_name=config.gemini_model_name,
    )

    st.session_state.page_documents = page_documents
    st.session_state.chunks = chunks
    st.session_state.vector_db = vector_db
    st.session_state.retriever = retriever
    st.session_state.rag_pipeline = rag_pipeline
    st.session_state.knowledge_base_ready = True

    st.success(
        f"Knowledge base ready: indexed {len(chunks)} chunks from "
        f"{len(saved_paths)} PDF file(s)."
    )


def render_retrieval_details(results: List[RetrievedChunk]) -> None:
    if not results:
        return

    best_score = max(result.combined_score for result in results)
    st.caption(f"Retrieval confidence: **{confidence_label(best_score)}**")

    with st.expander("Sources and retrieved context", expanded=False):
        for index, result in enumerate(results, start=1):
            citation = html.escape(format_citation(result.document))
            preview = html.escape(result.document.page_content[:900])
            st.markdown(
                f"""
                <div class="dm-source-card">
                    <div class="dm-source-topline">
                        <div class="dm-source-title">{index}. {citation}</div>
                        <div class="dm-score">score {result.combined_score:.3f}</div>
                    </div>
                    <p class="dm-source-text">
                        semantic {result.semantic_score:.3f} | keyword {result.keyword_score:.3f}
                    </p>
                    <p class="dm-source-text">{preview}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_empty_chat_state() -> None:
    st.markdown(
        """
        <div class="dm-empty">
            <p class="dm-empty-title">Start with a document-specific question</p>
            <p class="dm-empty-copy">
                Try asking for a definition, a comparison, a page reference, or an
                exam-style explanation after the knowledge base is built.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    examples = [
        "Explain the main idea of this document.",
        "List the important definitions with page references.",
        "Compare the two most important concepts.",
        "Create exam revision notes from the uploaded PDFs.",
    ]
    columns = st.columns(2)
    for index, example in enumerate(examples):
        with columns[index % 2]:
            st.caption(example)


def generate_summary(config: AppConfig) -> None:
    if not st.session_state.chunks:
        st.warning("Build a knowledge base before generating a summary.")
        return
    summarizer = DocumentSummarizer(
        gemini_api_key=config.gemini_api_key,
        model_name=config.gemini_model_name,
    )
    st.session_state.summary = summarizer.generate_summary(st.session_state.chunks)


def generate_flashcards(config: AppConfig) -> None:
    if not st.session_state.chunks:
        st.warning("Build a knowledge base before generating flashcards.")
        return
    generator = FlashcardGenerator(
        gemini_api_key=config.gemini_api_key,
        model_name=config.gemini_model_name,
    )
    st.session_state.flashcards = generator.generate_flashcards(st.session_state.chunks)


def clear_chat_history() -> None:
    st.session_state.messages = []


def main() -> None:
    st.set_page_config(
        page_title="DocuMind",
        page_icon="D",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    config = AppConfig.from_env()
    config.ensure_directories()
    initialize_session_state()
    inject_custom_css()
    render_header(config)

    with st.sidebar:
        st.header("Knowledge Base")
        render_sidebar_intro()
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        build_disabled = not uploaded_files
        if st.button(
            "Build Knowledge Base",
            type="primary",
            use_container_width=True,
            disabled=build_disabled,
        ):
            try:
                with st.spinner("Extracting, chunking, embedding, and indexing PDFs..."):
                    build_knowledge_base(uploaded_files or [], config)
            except Exception as exc:
                LOGGER.exception("Knowledge base build failed")
                st.error(f"Failed to build knowledge base: {exc}")

        st.divider()
        st.header("Study Tools")

        tool_disabled = not st.session_state.knowledge_base_ready
        if st.button("Generate Summary", use_container_width=True, disabled=tool_disabled):
            try:
                with st.spinner("Generating document summary..."):
                    generate_summary(config)
            except Exception as exc:
                LOGGER.exception("Summary generation failed")
                st.error(f"Failed to generate summary: {exc}")

        if st.button("Generate Flashcards", use_container_width=True, disabled=tool_disabled):
            try:
                with st.spinner("Generating flashcards..."):
                    generate_flashcards(config)
            except Exception as exc:
                LOGGER.exception("Flashcard generation failed")
                st.error(f"Failed to generate flashcards: {exc}")

        st.button("Clear Chat", use_container_width=True, on_click=clear_chat_history)

        st.divider()
        st.header("Index Stats")
        metric_left, metric_right = st.columns(2)
        metric_left.metric("Pages", len(st.session_state.page_documents))
        metric_right.metric("Chunks", len(st.session_state.chunks))
        st.caption(f"Chunking: {config.chunk_size} size, {config.chunk_overlap} overlap")
        st.caption(f"Embedding: {config.embedding_model_name.split('/')[-1]}")

    tab_chat, tab_summary, tab_flashcards = st.tabs(["Chat", "Summary", "Flashcards"])

    with tab_chat:
        if not config.gemini_api_key:
            st.info("Add GEMINI_API_KEY to your .env file before asking questions.")

        if not st.session_state.messages:
            render_empty_chat_state()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("retrieval_results"):
                    render_retrieval_details(message["retrieval_results"])

        question = st.chat_input("Ask a question about your uploaded documents")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                if not st.session_state.knowledge_base_ready:
                    answer = "Please upload PDFs and build the knowledge base first."
                    results: List[RetrievedChunk] = []
                    st.markdown(answer)
                elif not config.gemini_api_key:
                    answer = "Add GEMINI_API_KEY to your .env file before asking questions."
                    results = []
                    st.markdown(answer)
                else:
                    with st.spinner("Retrieving context and asking Gemini..."):
                        pipeline: RAGPipeline = st.session_state.rag_pipeline
                        response = pipeline.answer_question(
                            question=question,
                            chat_history=st.session_state.messages[:-1],
                        )
                    answer = response.answer
                    results = response.retrieved_chunks
                    st.markdown(answer)
                    render_retrieval_details(results)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "retrieval_results": results,
                }
            )

    with tab_summary:
        if st.session_state.summary:
            st.markdown(st.session_state.summary)
        else:
            st.markdown(
                """
                <div class="dm-empty">
                    <p class="dm-empty-title">No summary generated yet</p>
                    <p class="dm-empty-copy">
                        Build the knowledge base, then use Generate Summary from
                        the sidebar to create structured study notes.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_flashcards:
        if st.session_state.flashcards:
            st.markdown(st.session_state.flashcards)
        else:
            st.markdown(
                """
                <div class="dm-empty">
                    <p class="dm-empty-title">No flashcards generated yet</p>
                    <p class="dm-empty-copy">
                        Build the knowledge base, then use Generate Flashcards from
                        the sidebar to create question-answer revision cards.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
