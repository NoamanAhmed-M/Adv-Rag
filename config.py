"""
config.py — Zilliz Cloud only configuration.
Supports both .env (local) and Streamlit Cloud secrets (deployed).
"""
import os
from dotenv import load_dotenv

load_dotenv()

def _get(key: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then .env, then default."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

# ── Zilliz Cloud ──────────────────────────────────────────────────────────────
ZILLIZ_URI   = _get("ZILLIZ_URI")
ZILLIZ_TOKEN = _get("ZILLIZ_TOKEN")

# ── NVIDIA ────────────────────────────────────────────────────────────────────
NVIDIA_API_KEY   = _get("NVIDIA_API_KEY")
NVIDIA_API_URL   = "https://integrate.api.nvidia.com/v1/ocr"
NVIDIA_OCR_MODEL = "nvidia/nemoretriever-ocr-v1"

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_BACKEND = _get("OCR_BACKEND", "nvidia")

# ── App ───────────────────────────────────────────────────────────────────────
COLLECTION_NAME = _get("COLLECTION_NAME", "documents")
EMBEDDING_DIM   = int(_get("EMBEDDING_DIM", "4096"))
DATA_PATH       = "/tmp/rag_data"
LLM_MODEL       = _get("LLM_MODEL", "nvidia/nemotron-3-nano-30b-a3b")


def get_milvus_connection_params() -> dict:
    if not ZILLIZ_URI or not ZILLIZ_TOKEN:
        raise ValueError("ZILLIZ_URI or ZILLIZ_TOKEN is missing")
    return {
        "alias": "default",
        "uri":   ZILLIZ_URI,
        "token": ZILLIZ_TOKEN,
    }


def print_config():
    print("=" * 50)
    masked_uri = ZILLIZ_URI[:30] + "..." if ZILLIZ_URI else "NOT SET"
    print(f"  ZILLIZ_URI      : {masked_uri}")
    print(f"  ZILLIZ_TOKEN    : {'SET ✓' if ZILLIZ_TOKEN else 'NOT SET ✗'}")
    print(f"  OCR_BACKEND     : {OCR_BACKEND}")
    print(f"  COLLECTION_NAME : {COLLECTION_NAME}")
    print(f"  EMBEDDING_DIM   : {EMBEDDING_DIM}")
    print(f"  LLM_MODEL       : {LLM_MODEL}")
    print(f"  NVIDIA_API_KEY  : {'SET ✓' if NVIDIA_API_KEY else 'NOT SET ✗'}")
    print("=" * 50)