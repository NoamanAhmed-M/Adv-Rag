import os

# Try to get from Streamlit secrets first, then environment variables
try:
    import streamlit as st
    ZILLIZ_URI      = st.secrets.get("ZILLIZ_URI",      os.getenv("ZILLIZ_URI",      ""))
    ZILLIZ_TOKEN    = st.secrets.get("ZILLIZ_TOKEN",    os.getenv("ZILLIZ_TOKEN",    ""))
    NVIDIA_API_KEY  = st.secrets.get("NVIDIA_API_KEY",  os.getenv("NVIDIA_API_KEY",  ""))
    OCR_BACKEND     = st.secrets.get("OCR_BACKEND",     os.getenv("OCR_BACKEND",     "nvidia"))
    COLLECTION_NAME = st.secrets.get("COLLECTION_NAME", os.getenv("COLLECTION_NAME", "documents"))
    EMBEDDING_DIM   = int(st.secrets.get("EMBEDDING_DIM", os.getenv("EMBEDDING_DIM", "4096")))
    LLM_MODEL       = st.secrets.get("LLM_MODEL",       os.getenv("LLM_MODEL",       "nvidia/nemotron-3-nano-30b-a3b"))
    ADMIN_PASSWORD  = st.secrets.get("ADMIN_PASSWORD",  os.getenv("ADMIN_PASSWORD",  "admin123"))
    # nemoretriever-ocr-v1 is a dedicated OCR NIM — no model selector needed
    NVIDIA_OCR_MODEL = "nvidia/nemoretriever-ocr-v1"
except Exception:
    # Fallback to environment variables only
    ZILLIZ_URI      = os.getenv("ZILLIZ_URI",      "")
    ZILLIZ_TOKEN    = os.getenv("ZILLIZ_TOKEN",    "")
    NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY",  "")
    OCR_BACKEND     = os.getenv("OCR_BACKEND",     "nvidia")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
    EMBEDDING_DIM   = int(os.getenv("EMBEDDING_DIM", "4096"))
    LLM_MODEL       = os.getenv("LLM_MODEL",       "nvidia/nemotron-3-nano-30b-a3b")
    ADMIN_PASSWORD  = os.getenv("ADMIN_PASSWORD",  "admin123")
    NVIDIA_OCR_MODEL = "nvidia/nemoretriever-ocr-v1"

# ── Nvidia NIM ────────────────────────────────────────────────────────────────
# Hosted cloud endpoint for nemoretriever-ocr-v1
# ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr  (NOT integrate.api.nvidia.com)
NVIDIA_API_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr"

# DATA path (ephemeral on Streamlit Cloud — vectors live in Zilliz)
DATA_PATH = "/tmp/rag_data"


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
    masked_uri = (ZILLIZ_URI[:30] + "...") if ZILLIZ_URI else "NOT SET"
    print(f"  ZILLIZ_URI       : {masked_uri}")
    print(f"  ZILLIZ_TOKEN     : {'SET ✓' if ZILLIZ_TOKEN   else 'NOT SET ✗'}")
    print(f"  OCR_BACKEND      : {OCR_BACKEND}")
    print(f"  COLLECTION_NAME  : {COLLECTION_NAME}")
    print(f"  EMBEDDING_DIM    : {EMBEDDING_DIM}")
    print(f"  LLM_MODEL        : {LLM_MODEL}")
    print(f"  NVIDIA_API_KEY   : {'SET ✓' if NVIDIA_API_KEY else 'NOT SET ✗'}")
    print(f"  NVIDIA_API_URL   : {NVIDIA_API_URL}")
    print(f"  NVIDIA_OCR_MODEL : {NVIDIA_OCR_MODEL}")
    print(f"  ADMIN_PASSWORD   : {'SET ✓' if ADMIN_PASSWORD else 'NOT SET ✗'}")
    print("=" * 50)
