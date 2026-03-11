import streamlit as st
import os
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import logging
logging.getLogger('streamlit.watcher.local_sources_watcher').setLevel(logging.ERROR)

# Import config to get admin password
from config import ADMIN_PASSWORD

st.set_page_config(
    page_title="RAG Chat Interface",
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

# Initialize authentication state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

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


# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Authentication")
    
    # Password input (always show if not authenticated)
    if not st.session_state.authenticated:
        password = st.text_input("Enter admin password:", type="password")
        
        if st.button("Login", use_container_width=True):
            if password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.success("Authenticated successfully!")
                st.rerun()
            else:
                st.error("Incorrect password")
    else:
        st.success("Authenticated as Admin")
        
        # Logout button
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
    
    st.divider()
    
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
        )

    # ── Upload Documents Section (Protected) ─────────────────────────────────
    st.markdown("### Upload Documents")
    
    if not st.session_state.authenticated:
        st.warning("Login required to upload files")
        st.info("Please authenticate using the password field above to enable file uploads.")
        
        # Show disabled uploader (visual only)
        st.file_uploader(
            "Add files to knowledge base",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
            accept_multiple_files=True,
            disabled=True,
            help="Login required to upload files"
        )
    else:
        # Authenticated users can upload
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

    # ── Database Management Section (Protected) ─────────────────────────────
    st.markdown("### Database Management")
    
    if not st.session_state.authenticated:
        st.warning(" Login required for database operations")
        st.info("Please authenticate to clear the database.")
    else:
        # Clear DB button (only shown when authenticated)
        if st.button("Clear Database", type="secondary", use_container_width=True):
            try:
                from data_preprocessing import clear_database
            except ImportError as e:
                st.error(f"Import error: {str(e)}")
                st.info("This might be due to missing langchain_community. Installing required packages...")
                
                # Fallback: Try alternative import or provide instructions
                st.warning(
                    "The `langchain_community` module is missing. To fix this:\n\n"
                    "1. Add to requirements.txt:\n"
                    "   ```\n"
                    "   langchain-community>=0.2.0\n"
                    "   ```\n"
                    "2. Re-deploy the app\n\n"
                    "For now, database operations are disabled."
                )
            else:
                # Only runs if import was successful
                # Confirm deletion
                confirm = st.checkbox("I understand this will delete all documents and cannot be undone")
                if confirm:
                    with st.spinner("Clearing database..."):
                        try:
                            clear_database()
                            
                            # Also clear local files
                            if os.path.exists(DATA_PATH):
                                for file in os.listdir(DATA_PATH):
                                    file_path = os.path.join(DATA_PATH, file)
                                    if os.path.isfile(file_path):
                                        os.remove(file_path)
                            
                            st.success("Database cleared successfully!")
                            st.info("Re-upload your documents to rebuild.")
                            st.balloons()
                        except Exception as e:
                            st.error(f"Error clearing database: {str(e)}")
                else:
                    st.info("Please confirm to proceed with deletion")

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


# ── MAIN CHAT UI (outside sidebar) ─────────────────────────────────────────
st.title("RAG Document Chat")
st.caption("Retrieval-Augmented Generation over your documents")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            hal = message.get("hal_score")
            if hal == "yes":
                st.markdown("grounded in documents")
            elif hal == "no":
                st.markdown("may not be fully grounded")

            sources = message.get("sources", [])
            segments = message.get("segments", [])
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for i, src in enumerate(sources):
                        st.markdown(f"**Source:** `{src}`")
                        if i < len(segments) and segments[i]:
                            st.code(segments[i], language=None)

# Chat input
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
                    sources = result.source if isinstance(result.source, list) else [result.source]
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
                    "role": "assistant",
                    "content": display_text,
                    "sources": sources,
                    "segments": segments,
                    "hal_score": score_val,
                })

            except RuntimeError as e:
                # Collection doesn't exist yet — friendly message
                msg = str(e)
                st.warning(msg, icon="📭")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": msg,
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.error(traceback.format_exc())
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

st.divider()
st.caption("Built with Python · LangChain · Zilliz Cloud · Nvidia NIM")
