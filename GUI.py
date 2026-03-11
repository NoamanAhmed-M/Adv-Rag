import streamlit as st
import os
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import logging
logging.getLogger('streamlit.watcher.local_sources_watcher').setLevel(logging.ERROR)

from config import ADMIN_PASSWORD

# Pre-import Query at startup so the first prompt doesn't silently fail
# (lazy import inside the chat block causes a silent miss on the first Streamlit rerun)
try:
    from Query import query_rag as _query_rag_preload  # noqa: F401
except Exception:
    pass  # Will surface a proper error when the user queries

st.set_page_config(
    page_title="RAG Chat Interface",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* ── Dark industrial theme ── */
    .stApp {
        background-color: #0d0f12;
        color: #c8d0d9;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111318;
        border-right: 1px solid #1e2229;
    }
    [data-testid="stSidebar"] * {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.82rem !important;
    }

    /* Chat input */
    .stChatInput {
        position: fixed;
        bottom: 20px;
        width: 70%;
        left: 15%;
    }
    .stChatInput textarea {
        background-color: #161a1f !important;
        border: 1px solid #2a2f38 !important;
        color: #c8d0d9 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        border-radius: 8px !important;
    }
    .stChatInput textarea:focus {
        border-color: #4a9eff !important;
        box-shadow: 0 0 0 2px rgba(74,158,255,0.15) !important;
    }

    /* Chat messages */
    .stChatMessage {
        background-color: #13161b !important;
        border: 1px solid #1e2229 !important;
        border-radius: 10px !important;
        padding: 16px !important;
        margin-bottom: 10px !important;
    }

    /* Buttons */
    .stButton > button {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.05em !important;
        border-radius: 6px !important;
        border: 1px solid #2a2f38 !important;
        background-color: #161a1f !important;
        color: #8a95a3 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        border-color: #4a9eff !important;
        color: #4a9eff !important;
        background-color: rgba(74,158,255,0.08) !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #4a9eff !important;
        color: #0d0f12 !important;
        border-color: #4a9eff !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #6ab4ff !important;
        border-color: #6ab4ff !important;
        color: #0d0f12 !important;
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stTextInput > div > div > input[type="password"] {
        background-color: #161a1f !important;
        border: 1px solid #2a2f38 !important;
        color: #c8d0d9 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.82rem !important;
        border-radius: 6px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4a9eff !important;
        box-shadow: 0 0 0 2px rgba(74,158,255,0.15) !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #13161b !important;
        border: 1px dashed #2a2f38 !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #13161b !important;
        border: 1px solid #1e2229 !important;
        border-radius: 6px !important;
        color: #8a95a3 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.80rem !important;
    }
    .streamlit-expanderContent {
        background-color: #0f1217 !important;
        border: 1px solid #1e2229 !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
    }

    /* Alerts */
    .stAlert {
        border-radius: 6px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.78rem !important;
    }

    /* Divider */
    hr {
        border-color: #1e2229 !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #4a9eff !important;
    }

    /* Code blocks */
    code, pre {
        font-family: 'IBM Plex Mono', monospace !important;
        background-color: #0a0c0f !important;
        color: #7dd3fc !important;
        border: 1px solid #1e2229 !important;
        border-radius: 4px !important;
    }

    /* Title */
    h1 {
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 500 !important;
        letter-spacing: -0.02em !important;
        color: #e8edf2 !important;
    }
    h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #c8d0d9 !important;
    }

    /* Caption */
    .stCaption, small {
        color: #4a5260 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem !important;
    }

    /* Checkbox */
    .stCheckbox label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.78rem !important;
        color: #8a95a3 !important;
    }

    /* Selectbox / other widgets label */
    .stMarkdown p {
        color: #c8d0d9;
        line-height: 1.65;
    }

    /* Section headers in sidebar */
    .sidebar-section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #4a5260;
        padding: 12px 0 4px 0;
        border-bottom: 1px solid #1e2229;
        margin-bottom: 10px;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.70rem;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 500;
        letter-spacing: 0.06em;
    }
    .status-badge.connected {
        background: rgba(34,197,94,0.1);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.2);
    }
    .status-badge.disconnected {
        background: rgba(239,68,68,0.1);
        color: #f87171;
        border: 1px solid rgba(239,68,68,0.2);
    }
    .status-badge.admin {
        background: rgba(74,158,255,0.1);
        color: #4a9eff;
        border: 1px solid rgba(74,158,255,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

# ── Data path ─────────────────────────────────────────────────────────────────
DATA_PATH = "/tmp/rag_data"
os.makedirs(DATA_PATH, exist_ok=True)


def process_uploaded_files(uploaded_files):
    """Save uploaded files to DATA_PATH then index them into Zilliz Cloud."""
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


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Auth ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-header">Authentication</div>', unsafe_allow_html=True)

    if not st.session_state.authenticated:
        password = st.text_input("Admin password", type="password", label_visibility="collapsed",
                                 placeholder="Enter admin password...")
        if st.button("→ Login", use_container_width=True):
            if password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
    else:
        st.markdown('<span class="status-badge admin">● ADMIN</span>', unsafe_allow_html=True)
        st.markdown("")
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    st.divider()

    # ── Config / status ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-header">System Status</div>', unsafe_allow_html=True)

    from config import ZILLIZ_URI, ZILLIZ_TOKEN
    if ZILLIZ_URI and ZILLIZ_TOKEN:
        st.markdown('<span class="status-badge connected">● Zilliz Cloud</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge disconnected">✕ Zilliz — credentials missing</span>', unsafe_allow_html=True)

    from config import OCR_BACKEND
    if OCR_BACKEND in ("ollama", "both"):
        st.warning(
            "OCR backend includes 'ollama' — this won't work on cloud. "
            "Set `OCR_BACKEND=nvidia` in your environment.",
        )

    st.divider()

    # ── Upload Documents ──────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-header">Upload Documents</div>', unsafe_allow_html=True)

    if not st.session_state.authenticated:
        st.caption("🔒 Login required to upload files")
        st.file_uploader(
            "Add files to knowledge base",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
            accept_multiple_files=True,
            disabled=True,
            label_visibility="collapsed",
            help="Login required"
        )
    else:
        uploaded_files = st.file_uploader(
            "PDF or image files",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            if st.button("Add to Database", type="primary", use_container_width=True):
                with st.spinner("Processing…"):
                    try:
                        saved_files, chunk_count = process_uploaded_files(uploaded_files)
                        st.success(f"{len(saved_files)} file(s) indexed")
                        st.info(f"→ {chunk_count} chunks stored")
                        with st.expander("Processed files"):
                            for f in saved_files:
                                st.text(f"• {f}")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.error(traceback.format_exc())

    st.divider()

    # ── Database Management ───────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-header">Database</div>', unsafe_allow_html=True)

    if not st.session_state.authenticated:
        st.caption("🔒 Login required for database operations")
    else:
        # Step 1 — show the "Clear Database" button
        if not st.session_state.confirm_clear:
            if st.button("Clear Database", type="secondary", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()

        # Step 2 — confirmation dialog
        else:
            st.warning("⚠️ This will permanently delete all vectors from Zilliz Cloud.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✓ Confirm Delete", type="primary", use_container_width=True):
                    with st.spinner("Clearing database…"):
                        try:
                            # Safe import — isolate from top-level ollama import in data_preprocessing
                            import importlib, sys

                            # Temporarily stub 'ollama' if missing so the module loads
                            if "ollama" not in sys.modules:
                                from unittest.mock import MagicMock
                                sys.modules["ollama"] = MagicMock()

                            if "data_preprocessing" in sys.modules:
                                dp = sys.modules["data_preprocessing"]
                            else:
                                import data_preprocessing as dp

                            dp.clear_database()

                            # Also wipe local temp files
                            if os.path.exists(DATA_PATH):
                                for file in os.listdir(DATA_PATH):
                                    fp = os.path.join(DATA_PATH, file)
                                    if os.path.isfile(fp):
                                        os.remove(fp)

                            st.session_state.confirm_clear = False
                            st.success("✓ Database cleared successfully.")
                            st.balloons()
                            st.rerun()
                        except Exception as e:
                            st.session_state.confirm_clear = False
                            st.error(f"Error: {str(e)}")
                            st.error(traceback.format_exc())
            with col2:
                if st.button("✕ Cancel", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()

    # ── Files in data folder (auth-gated) ─────────────────────────────────────
    with st.expander("Files in Data Folder"):
        if not st.session_state.authenticated:
            st.caption("🔒 Login to view stored files")
        else:
            existing_files = sorted(os.listdir(DATA_PATH)) if os.path.exists(DATA_PATH) else []
            if existing_files:
                st.write(f"**{len(existing_files)} file(s)**")
                for f in existing_files:
                    st.text(f"• {f}")
            else:
                st.caption("No files yet. Upload above to get started.")

    st.divider()
    st.caption("Zilliz Cloud · Nvidia NIM · Nemotron · LangChain")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CHAT
# ═════════════════════════════════════════════════════════════════════════════
st.title("RAG Document Chat")
st.caption("Retrieval-Augmented Generation over your documents")

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            hal = message.get("hal_score")
            sources  = message.get("sources", [])
            segments = message.get("segments", [])

            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    if hal == "yes":
                        st.success("Grounded in documents")
                    elif hal == "no":
                        st.warning("May not be fully grounded")

                    for i, src in enumerate(sources):
                        st.markdown(f"**Source:** `{src}`")
                        if i < len(segments) and segments[i]:
                            st.code(segments[i], language=None)
                        if i < len(sources) - 1:
                            st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    # 1. Save user message immediately so it always appears
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Run the query and save result — NO st.rerun() anywhere here.
    #    The history loop above will render everything on the next natural Streamlit cycle.
    with st.spinner("Searching documents…"):
        try:
            from Query import query_rag
            answer_text, result, hal_score = query_rag(prompt)

            # Resolve display text
            if isinstance(answer_text, str) and answer_text:
                display_text = answer_text
            elif hasattr(result, "content"):
                display_text = result.content
            elif isinstance(result, str):
                display_text = result
            else:
                display_text = str(result)

            # Resolve score
            if hasattr(hal_score, "binary_score"):
                score_val = hal_score.binary_score
            elif isinstance(hal_score, str):
                score_val = hal_score
            else:
                score_val = "not_applicable"

            # Resolve sources / segments
            sources, segments = [], []
            if hasattr(result, "source"):
                sources = result.source if isinstance(result.source, list) else [result.source]
            if hasattr(result, "segment"):
                segments = result.segment if isinstance(result.segment, list) else [result.segment]

            st.session_state.messages.append({
                "role":      "assistant",
                "content":   display_text,
                "sources":   sources,
                "segments":  segments,
                "hal_score": score_val,
            })

        except RuntimeError as e:
            st.session_state.messages.append({
                "role":    "assistant",
                "content": str(e),
            })

        except Exception as e:
            st.session_state.messages.append({
                "role":    "assistant",
                "content": f"Error: {str(e)}

{traceback.format_exc()}",
            })

    # 3. Single rerun AFTER everything is saved — renders the full history cleanly
    st.rerun()

st.divider()
st.caption("Built with Python · LangChain · Zilliz Cloud · Nvidia NIM")
