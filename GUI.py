import streamlit as st
import os
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import logging
logging.getLogger('streamlit.watcher.local_sources_watcher').setLevel(logging.ERROR)

st.set_page_config(
    page_title="RAG Chat Interface",
    page_icon="",
    layout="wide"
)

st.markdown("""
    <style>
    .stChatInput {
        position: fixed;
        bottom: 20px;
        width: 70%;
        left: 15%;
    }
    .stChatMessage {
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Data path: /tmp staging area (wiped on restart — vectors live in Zilliz Cloud)
DATA_PATH = "/tmp/rag_data"
os.makedirs(DATA_PATH, exist_ok=True)


def process_uploaded_files(uploaded_files):
    """Save uploaded files to Data/ then index them into Zilliz Cloud."""
    from data_preprocessing import load_documents, split_document, add_to_milvus

    saved_files = []
    for uploaded_file in uploaded_files:
        dest = os.path.join(DATA_PATH, uploaded_file.name)
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(uploaded_file.name)

    documents = load_documents(DATA_PATH)
    chunks    = split_document(documents)
    add_to_milvus(chunks)

    return saved_files, len(chunks)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")

    # ── Cloud status indicator ────────────────────────────────────────────────
    from config import ZILLIZ_URI, ZILLIZ_TOKEN
    if ZILLIZ_URI and ZILLIZ_TOKEN:
        st.success("Zilliz Cloud connected")
    else:
        st.error("Zilliz credentials missing in .env")

    st.divider()

    # ── OCR backend warning ───────────────────────────────────────────────────
    from config import OCR_BACKEND
    if OCR_BACKEND in ("ollama", "both"):
        st.warning(
            "OCR backend includes 'ollama'. "
            "Ollama requires a local server and **will not work on cloud**. "
            "Set `OCR_BACKEND=nvidia` in your environment variables.",
            icon=""
        )

    st.markdown("### Upload Documents")

    uploaded_files = st.file_uploader(
        "Add files to knowledge base",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
        accept_multiple_files=True,
        help="Supported: PDF and image files"
    )

    if uploaded_files:
        if st.button("Add to Database", type="primary", use_container_width=True):
            with st.spinner("Processing documents..."):
                try:
                    saved_files, chunk_count = process_uploaded_files(uploaded_files)
                    st.success(f"{len(saved_files)} file(s) added")
                    st.info(f"→ {chunk_count} chunks indexed")
                    with st.expander("Processed files"):
                        for f in saved_files:
                            st.text(f"• {f}")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.error(traceback.format_exc())

    st.divider()

    st.markdown("### Database Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Reindex", type="secondary", use_container_width=True):
            with st.spinner("Reindexing..."):
                try:
                    from data_preprocessing import main
                    main()
                    st.success("Done!")
                except Exception as e:
                    st.error(str(e))

    with col2:
        if st.button("Clear DB", type="secondary", use_container_width=True):
            try:
                from data_preprocessing import clear_database
                clear_database()
                st.success("DB cleared!")
                st.info("Re-upload your documents to rebuild.")
            except Exception as e:
                st.error(str(e))

    with st.expander("Files in Data folder"):
        existing_files = sorted(os.listdir(DATA_PATH)) if os.path.exists(DATA_PATH) else []
        if existing_files:
            st.write(f"**{len(existing_files)} file(s)**")
            for f in existing_files:
                st.text(f"• {f}")
        else:
            st.caption("No files yet. Upload above to get started.")

    st.divider()
    st.markdown("**Stack**")
    st.caption("Zilliz Cloud · Nvidia NIM · Nemotron · LangChain")
    st.caption("Duplicate chunks skipped automatically (source:page:chunk_index)")


# ── Main chat UI ──────────────────────────────────────────────────────────────
st.title("RAG Document Chat")
st.caption("Retrieval-Augmented Generation over your documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            hal = message.get("hal_score")
            if hal == "yes":
                st.markdown('<span class="hal-badge-ok"> grounded in documents</span>', unsafe_allow_html=True)
            elif hal == "no":
                st.markdown('<span class="hal-badge-warn"> may not be fully grounded</span>', unsafe_allow_html=True)

            sources  = message.get("sources", [])
            segments = message.get("segments", [])
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for i, src in enumerate(sources):
                        st.markdown(f"**Source:** `{src}`")
                        if i < len(segments) and segments[i]:
                            st.code(segments[i], language=None)


if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                from Query import query_rag
                answer_text, result, hal_score = query_rag(prompt)

                if isinstance(answer_text, str) and answer_text:
                    display_text = answer_text
                elif hasattr(result, "content"):
                    display_text = result.content
                elif isinstance(result, str):
                    display_text = result
                else:
                    display_text = str(result)

                st.markdown(display_text)

                if hasattr(hal_score, "binary_score"):
                    score_val = hal_score.binary_score
                elif isinstance(hal_score, str):
                    score_val = hal_score
                else:
                    score_val = "not_applicable"

                sources, segments = [], []
                if hasattr(result, "source"):
                    sources  = result.source  if isinstance(result.source,  list) else [result.source]
                if hasattr(result, "segment"):
                    segments = result.segment if isinstance(result.segment, list) else [result.segment]

                with st.expander(f"Sources ({len(sources)})"):
                    if score_val == "yes":
                        st.success("Grounded in documents")
                    elif score_val == "no":
                        st.warning("May not be fully grounded")

                    for i, src in enumerate(sources):
                        st.markdown(f"**Source:** `{os.path.basename(src)}`")
                        st.caption(f"Full path: {src}")
                        if i < len(segments) and segments[i]:
                            st.markdown("**Retrieved Segment:**")
                            st.code(segments[i], language=None)
                        if i < len(sources) - 1:
                            st.divider()

                st.session_state.messages.append({
                    "role":     "assistant",
                    "content":  display_text,
                    "sources":  sources,
                    "segments": segments,
                    "hal_score": score_val,
                })

            except RuntimeError as e:
                # Collection doesn't exist yet — friendly message
                msg = str(e)
                st.warning(msg, icon="📭")
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": msg,
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.error(traceback.format_exc())
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": error_msg,
                })

st.divider()
st.caption("Built with Python · LangChain · Zilliz Cloud · Nvidia NIM")