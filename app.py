"""Polished Streamlit interface for the GenAI RAG application."""

from __future__ import annotations

import hashlib
import uuid

import streamlit as st

from rag_pipeline import (
    delete_collection_if_exists,
    generate_answer,
    retrieve_relevant_chunks,
    store_chunks_in_chromadb,
)
from utils.pdf_processor import create_chunks, extract_text_from_pdf


st.set_page_config(
    page_title="GenAI RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Small CSS layer for a cleaner, product-style dashboard without external assets.
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1450px;}
      .hero {padding: 1.35rem 1.6rem; border-radius: 18px; margin-bottom: 1rem;
             background: linear-gradient(115deg, rgba(75, 91, 220, .16), rgba(34, 197, 170, .10));
             border: 1px solid rgba(128, 128, 128, .22);}
      .hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
      .hero p {margin: 0; color: var(--text-color); opacity: .82;}
      div[data-testid="stMetric"] {padding: .8rem 1rem; border: 1px solid rgba(128,128,128,.22);
             border-radius: 14px; background: rgba(128,128,128,.035);}
      div[data-testid="stMetricLabel"] p {font-size: .88rem;}
      .section-note {opacity: .78; font-size: .92rem;}
      div.stButton > button[kind="primary"] {min-height: 2.8rem; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def process_pdf(pdf_bytes: bytes, chunk_size: int, chunk_overlap: int):
    """Cache extraction and chunking so ordinary UI reruns stay fast."""
    pages = extract_text_from_pdf(pdf_bytes)
    chunks = create_chunks(pages, chunk_size=chunk_size, overlap=chunk_overlap)
    return pages, chunks


def initialize_session_state() -> None:
    defaults = {
        "rag_session_id": uuid.uuid4().hex[:12],
        "active_pdf_hash": None,
        "active_collection_name": None,
        "indexed_pdf_hash": None,
        "indexed_filename": None,
        "indexed_chunk_size": None,
        "indexed_chunk_overlap": None,
        "last_question": "",
        "last_answer": "",
        "last_sources": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def collection_name_for_session(session_id: str, pdf_hash: str) -> str:
    """Return a valid, session-specific Chroma collection name."""
    return f"rag_{session_id}_{pdf_hash[:12]}"


def reset_answer() -> None:
    st.session_state.last_question = ""
    st.session_state.last_answer = ""
    st.session_state.last_sources = []


initialize_session_state()

with st.sidebar:
    st.header("⚙️ Document settings")
    st.caption("These settings apply when the PDF is split into chunks.")
    chunk_size = int(
        st.number_input(
            "Chunk size (words)", min_value=100, max_value=1000,
            value=500, step=50,
            help="Approximate maximum number of words in each chunk.",
        )
    )
    chunk_overlap = int(
        st.number_input(
            "Chunk overlap (words)", min_value=0, max_value=400,
            value=100, step=25,
            help="Words repeated between adjacent chunks to preserve context.",
        )
    )
    retrieval_count = int(
        st.slider("Relevant chunks to retrieve", min_value=1, max_value=10, value=5)
    )
    model_name = st.text_input(
        "Groq model", value="openai/gpt-oss-20b",
        help="Hosted model ID used by the Groq API. Requires GROQ_API_KEY in app secrets.",
    ).strip()
    st.divider()
    st.caption("PDFs are processed by this app server; answer generation uses the configured Groq API.")

st.markdown(
    """
    <div class="hero">
      <h1>📚 GenAI RAG</h1>
      <p>Chat with your PDF using semantic search, vector embeddings, ChromaDB, and a hosted LLM.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if chunk_overlap >= chunk_size:
    st.error("Chunk overlap must be smaller than chunk size. Adjust the values in Document settings.")
    st.stop()

st.subheader("1. Upload your document")
uploaded_file = st.file_uploader(
    "Choose a text-based PDF to get started", type=["pdf"],
    help="Scanned/image-only PDFs need OCR, which is not included in this version. Uploads are limited to 20 MB.",
)

if uploaded_file is None:
    st.info("Upload a PDF to see document statistics, build its index, and ask questions.")
    st.markdown("**Workflow**　① Upload PDF　→　② Review document　→　③ Build index　→　④ Ask questions")
    st.stop()

pdf_bytes = uploaded_file.getvalue()
pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

# A new file gets a clean session index and clears answers from the previous file.
if st.session_state.active_pdf_hash != pdf_hash:
    previous_collection = st.session_state.active_collection_name
    if previous_collection:
        try:
            delete_collection_if_exists(previous_collection)
        except Exception:
            pass
    st.session_state.active_pdf_hash = pdf_hash
    st.session_state.active_collection_name = None
    st.session_state.indexed_pdf_hash = None
    st.session_state.indexed_filename = None
    st.session_state.indexed_chunk_size = None
    st.session_state.indexed_chunk_overlap = None
    reset_answer()

try:
    with st.spinner("Reading the PDF and preparing text chunks…"):
        pages, chunks = process_pdf(pdf_bytes, chunk_size, chunk_overlap)
except Exception as exc:
    st.error(f"Could not read this PDF. It may be damaged or password-protected. Details: {exc}")
    st.stop()

if not pages or not chunks:
    st.warning("No selectable text was found. This may be a scanned PDF; OCR is not included in this version.")
    st.stop()

# If chunking settings change after indexing, ask the user to rebuild the index.
is_indexed = (
    st.session_state.indexed_pdf_hash == pdf_hash
    and st.session_state.active_collection_name is not None
    and st.session_state.indexed_chunk_size == chunk_size
    and st.session_state.indexed_chunk_overlap == chunk_overlap
)

if st.session_state.indexed_pdf_hash == pdf_hash and not is_indexed:
    st.session_state.last_answer = ""
    st.session_state.last_sources = []

word_count = sum(len(page["text"].split()) for page in pages)
page_count = len(pages)
chunk_count = len(chunks)

st.subheader("2. Document overview")
info_col1, info_col2 = st.columns([3, 1])
with info_col1:
    st.markdown(f"**📄 {uploaded_file.name}**")
    st.caption(f"{uploaded_file.size / 1024:,.1f} KB · {page_count} pages with extractable text")
with info_col2:
    if is_indexed:
        st.success("Index ready", icon="✅")
    else:
        st.info("Ready to index", icon="🧠")

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Pages processed", f"{page_count:,}")
metric2.metric("Words extracted", f"{word_count:,}")
metric3.metric("Text chunks", f"{chunk_count:,}")
metric4.metric("Index status", "Ready" if is_indexed else "Not indexed")

# Keep the primary action above the detail tabs, so users never need to scroll through chunks.
st.subheader("3. Build your searchable index")
index_col, status_col = st.columns([1, 2], vertical_alignment="center")
with index_col:
    index_clicked = st.button(
        "🧠 Generate embeddings & index PDF",
        type="primary", use_container_width=True,
        help="Creates embeddings for the chunks and stores them in ChromaDB.",
    )
with status_col:
    if is_indexed:
        st.success(f"This document is indexed ({chunk_count:,} chunks). You can ask questions below.")
    else:
        st.markdown(
            "<div class='section-note'>Indexing converts the chunks into vector embeddings and stores them "
            "in ChromaDB. This step is required before question answering.</div>",
            unsafe_allow_html=True,
        )

if index_clicked:
    new_collection_name = collection_name_for_session(st.session_state.rag_session_id, pdf_hash)
    try:
        with st.spinner("Loading the embedding model, generating vectors, and saving the index…"):
            collection = store_chunks_in_chromadb(
                chunks, new_collection_name, source_name=uploaded_file.name
            )
        st.session_state.active_collection_name = collection.name
        st.session_state.indexed_pdf_hash = pdf_hash
        st.session_state.indexed_filename = uploaded_file.name
        st.session_state.indexed_chunk_size = chunk_size
        st.session_state.indexed_chunk_overlap = chunk_overlap
        reset_answer()
        st.success(f"Successfully indexed {collection.count():,} chunks. Open **Ask Questions** to query the PDF.")
        st.rerun()
    except Exception as exc:
        st.error(
            "Indexing failed. Check that the embedding model can be loaded and required packages are installed. "
            f"Details: {exc}"
        )

# Re-evaluate status after the index action may have rerun the app.
is_indexed = (
    st.session_state.indexed_pdf_hash == pdf_hash
    and st.session_state.active_collection_name is not None
    and st.session_state.indexed_chunk_size == chunk_size
    and st.session_state.indexed_chunk_overlap == chunk_overlap
)

st.divider()
tab_overview, tab_questions, tab_text, tab_chunks = st.tabs(
    ["📊 Overview", "💬 Ask Questions", "📄 Extracted Text", "🧩 Chunks"]
)

with tab_overview:
    st.markdown("### Processing summary")
    st.write(
        "Your PDF has been read and split into overlapping text chunks. The chunks remain available to the RAG "
        "pipeline; this dashboard only shows their count here so the main actions stay easy to find."
    )
    overview_left, overview_right = st.columns(2)
    with overview_left:
        st.markdown("**Pipeline status**")
        st.markdown(
            f"- {'✅' if pages else '⏳'} PDF text extraction\n"
            f"- {'✅' if chunks else '⏳'} Chunk generation\n"
            f"- {'✅' if is_indexed else '⬜'} Embeddings and ChromaDB index\n"
            f"- {'✅' if is_indexed else '⬜'} Question answering available"
        )
    with overview_right:
        st.markdown("**Current configuration**")
        st.markdown(
            f"- Chunk size: **{chunk_size} words**\n"
            f"- Chunk overlap: **{chunk_overlap} words**\n"
            f"- Retrieval results: **{retrieval_count}**\n"
            f"- Groq model: **{model_name or 'Not specified'}**"
        )
    if not is_indexed:
        st.info("Next step: select **Generate embeddings & index PDF** above to enable question answering.")
    else:
        st.success("Everything is ready. Open the **Ask Questions** tab to chat with your document.")

with tab_questions:
    st.markdown("### Ask your document")
    st.caption("Answers are generated from retrieved PDF passages. Check the cited pages for important details.")
    if not is_indexed:
        st.info("Build the searchable index using the button above before asking a question.")
    else:
        with st.form("question_form", clear_on_submit=False):
            question_input = st.text_area(
                "Your question",
                value=st.session_state.last_question,
                placeholder="Example: What are the main ideas discussed in this document?",
                height=100,
            )
            ask_clicked = st.form_submit_button("🔎 Retrieve context & generate answer", type="primary")
        if ask_clicked:
            if not question_input.strip():
                st.warning("Enter a question first.")
            elif not model_name:
                st.warning("Enter a Groq model ID in the sidebar.")
            else:
                try:
                    with st.spinner("Searching the document for relevant passages…"):
                        sources = retrieve_relevant_chunks(
                            question_input.strip(),
                            st.session_state.active_collection_name,
                            top_k=retrieval_count,
                        )
                    if not sources:
                        st.warning("No relevant chunks were retrieved from this document.")
                    else:
                        with st.spinner(f"Generating an answer with Groq ({model_name})…"):
                            answer = generate_answer(question_input.strip(), sources, model_name=model_name)
                        st.session_state.last_question = question_input.strip()
                        st.session_state.last_answer = answer
                        st.session_state.last_sources = sources
                except Exception as exc:
                    st.error(
                        "Could not retrieve context or generate an answer. Check your Groq API key, model ID, and network connection. "
                        f"Details: {exc}"
                    )
        if st.session_state.last_answer:
            st.markdown("#### Answer")
            st.markdown(st.session_state.last_answer)
            st.caption("Generated from retrieved PDF context. Verify important details against the source document.")
            with st.expander(f"📌 Retrieved source passages ({len(st.session_state.last_sources)})", expanded=False):
                for rank, item in enumerate(st.session_state.last_sources, start=1):
                    similarity = item.get("similarity")
                    similarity_label = f" · similarity {similarity:.3f}" if similarity is not None else ""
                    with st.expander(
                        f"Result {rank} · page {item['page_number']} · chunk {item['chunk_id']}{similarity_label}"
                    ):
                        st.write(item["text"])
                        st.caption(f"Source: {item.get('source', uploaded_file.name)}")

with tab_text:
    st.markdown("### Extracted PDF text")
    st.caption("Select a page to inspect the text extracted from the original PDF.")
    page_numbers = [page["page_number"] for page in pages]
    selected_page_number = st.selectbox(
        "Choose a page", page_numbers,
        format_func=lambda number: f"Page {number}",
        key=f"selected_page_{pdf_hash[:8]}",
    )
    selected_page = next(page for page in pages if page["page_number"] == selected_page_number)
    st.text_area(
        "Page text", value=selected_page["text"], height=380,
        key=f"page_text_{pdf_hash[:8]}_{selected_page_number}", disabled=True,
    )
    st.caption(f"Words on this page: {len(selected_page['text'].split()):,}")

with tab_chunks:
    st.markdown("### Chunk inspector")
    st.caption("Chunks are generated for retrieval. Choose one to inspect its text and metadata; all chunks are still used during indexing.")
    chunk_ids = list(range(len(chunks)))
    selected_chunk_index = st.selectbox(
        f"Choose a chunk ({len(chunks):,} total)",
        chunk_ids,
        format_func=lambda idx: f"Chunk {chunks[idx]['chunk_id']} · page {chunks[idx]['page_number']}",
        key=f"selected_chunk_{pdf_hash[:8]}_{chunk_size}_{chunk_overlap}",
    )
    selected_chunk = chunks[selected_chunk_index]
    st.text_area(
        "Chunk text", value=selected_chunk["text"], height=300,
        key=f"chunk_text_{pdf_hash[:8]}_{selected_chunk['chunk_id']}", disabled=True,
    )
    meta1, meta2, meta3 = st.columns(3)
    meta1.metric("Chunk ID", selected_chunk["chunk_id"])
    meta2.metric("Source page", selected_chunk["page_number"])
    meta3.metric("Words in chunk", f"{len(selected_chunk['text'].split()):,}")

st.divider()
st.caption("GenAI RAG · Streamlit · Sentence Transformers · ChromaDB · Groq API · PDF question answering")
