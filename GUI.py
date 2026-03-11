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
    page_icon="📚",
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


# ── Authentication UI ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Authentication")
    
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
        st.success(" Authenticated as Admin")
        
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
        st.warning(" Login required to upload files")
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
        st.warning("Login required for database operations")
        st.info("Please authenticate to clear the database.")
    else:
        # Clear DB button (only shown when authenticated)
        if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
            try:
                # Try to import the clear_database function
                try:
                    from data_preprocessing import clear_database
                except ImportError as e:
                    st.error(f"Import error: {str(e)}")
                    st.info("This might be due to missing langchain_community. Installing required packages...")
                    
                    # Fallback: Try alternative import or provide instructions
                    st.warning("""
                    The `langchain_community` module is missing. To fix this:
                    
                    1. Add to requirements.txt:
