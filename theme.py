"""Shared Streamlit theme, CSS injection, and default session-state initialiser."""
import streamlit as st

# ── Default pipeline settings ─────────────────────────────────────────────────
DEFAULTS = {
    # Pipeline toggles
    "use_query_rewrite":      True,
    "use_relevancy_check":    True,
    "use_reranker":           True,
    "use_hallucination_check":True,
    # Retrieval
    "top_k":                  10,
    "rerank_top_n":           5,
    "score_threshold":        0.25,
    # Chunking
    "chunking_mode":          "standard",
    "chunk_size":             800,
    "chunk_overlap":          80,
    "prop_pre_chunk_size":    1000,
    "prop_pre_chunk_overlap": 100,
    "prop_min_quality_score": 6,
    "prop_quality_check":     True,
    # Generation
    "llm_temperature":        0.0,
    "max_answer_sentences":   6,
    # Display
    "show_raw_sources":       True,
    "show_hallucination_badge":True,
}


def init_settings():
    """Initialise all settings keys in session_state once."""
    for key, val in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val
    # Auth / misc
    for key, val in [
        ("authenticated", False),
        ("messages", []),
        ("confirm_clear", False),
        ("uploaded_file_names", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = val


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

/* ── Dark industrial theme ── */
.stApp                      { background-color: #0d0f12; color: #c8d0d9; }
[data-testid="stSidebar"]   { background-color: #111318; border-right: 1px solid #1e2229; }
[data-testid="stSidebar"] * { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.82rem !important; }

/* Nav tabs */
[data-testid="stSidebarNav"] a { color: #8a95a3 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.80rem !important; }
[data-testid="stSidebarNav"] a:hover { color: #4a9eff !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] { color: #4a9eff !important; border-left: 2px solid #4a9eff; padding-left: 6px; }

/* Chat input */
.stChatInput { position: fixed; bottom: 20px; width: 70%; left: 15%; }
.stChatInput textarea { background-color: #161a1f !important; border: 1px solid #2a2f38 !important; color: #c8d0d9 !important; font-family: 'IBM Plex Sans', sans-serif !important; border-radius: 8px !important; }
.stChatInput textarea:focus { border-color: #4a9eff !important; box-shadow: 0 0 0 2px rgba(74,158,255,0.15) !important; }

/* Chat messages */
.stChatMessage { background-color: #13161b !important; border: 1px solid #1e2229 !important; border-radius: 10px !important; padding: 16px !important; margin-bottom: 10px !important; }

/* Buttons */
.stButton > button { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important; letter-spacing: 0.05em !important; border-radius: 6px !important; border: 1px solid #2a2f38 !important; background-color: #161a1f !important; color: #8a95a3 !important; transition: all 0.2s ease !important; }
.stButton > button:hover { border-color: #4a9eff !important; color: #4a9eff !important; background-color: rgba(74,158,255,0.08) !important; }
.stButton > button[kind="primary"] { background-color: #4a9eff !important; color: #0d0f12 !important; border-color: #4a9eff !important; font-weight: 600 !important; }
.stButton > button[kind="primary"]:hover { background-color: #6ab4ff !important; border-color: #6ab4ff !important; color: #0d0f12 !important; }

/* Inputs */
.stTextInput > div > div > input,
.stTextInput > div > div > input[type="password"] { background-color: #161a1f !important; border: 1px solid #2a2f38 !important; color: #c8d0d9 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.82rem !important; border-radius: 6px !important; }
.stTextInput > div > div > input:focus { border-color: #4a9eff !important; box-shadow: 0 0 0 2px rgba(74,158,255,0.15) !important; }

/* Slider */
.stSlider > div { color: #c8d0d9; }
div[data-baseweb="slider"] div[role="slider"] { background-color: #4a9eff !important; }

/* Selectbox */
div[data-baseweb="select"] > div { background-color: #161a1f !important; border-color: #2a2f38 !important; color: #c8d0d9 !important; }

/* Number input */
input[type="number"] { background-color: #161a1f !important; border-color: #2a2f38 !important; color: #c8d0d9 !important; }

/* Toggle / checkbox */
.stCheckbox label, .stToggle label { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important; color: #8a95a3 !important; }

/* File uploader */
[data-testid="stFileUploader"] { background-color: #13161b !important; border: 1px dashed #2a2f38 !important; border-radius: 8px !important; padding: 8px !important; }

/* Expander */
.streamlit-expanderHeader { background-color: #13161b !important; border: 1px solid #1e2229 !important; border-radius: 6px !important; color: #8a95a3 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.80rem !important; }
.streamlit-expanderContent { background-color: #0f1217 !important; border: 1px solid #1e2229 !important; border-top: none !important; border-radius: 0 0 6px 6px !important; }

/* Alerts */
.stAlert { border-radius: 6px !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important; }

hr { border-color: #1e2229 !important; }
.stSpinner > div { border-top-color: #4a9eff !important; }
code, pre { font-family: 'IBM Plex Mono', monospace !important; background-color: #0a0c0f !important; color: #7dd3fc !important; border: 1px solid #1e2229 !important; border-radius: 4px !important; }

h1 { font-family: 'IBM Plex Mono', monospace !important; font-weight: 500 !important; letter-spacing: -0.02em !important; color: #e8edf2 !important; }
h2, h3 { font-family: 'IBM Plex Mono', monospace !important; color: #c8d0d9 !important; }

.stCaption, small { color: #4a5260 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.72rem !important; }
.stMarkdown p { color: #c8d0d9; line-height: 1.65; }

.sidebar-section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; font-weight: 500;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #4a5260; padding: 12px 0 4px 0;
    border-bottom: 1px solid #1e2229; margin-bottom: 10px;
}

.status-badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.70rem; font-family: 'IBM Plex Mono', monospace; font-weight: 500; letter-spacing: 0.06em; }
.status-badge.connected   { background: rgba(34,197,94,0.1);  color: #4ade80; border: 1px solid rgba(34,197,94,0.2); }
.status-badge.disconnected{ background: rgba(239,68,68,0.1);  color: #f87171; border: 1px solid rgba(239,68,68,0.2); }
.status-badge.admin       { background: rgba(74,158,255,0.1); color: #4a9eff; border: 1px solid rgba(74,158,255,0.2); }
.status-badge.warning     { background: rgba(251,191,36,0.1); color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }

/* Settings page cards */
.setting-card {
    background-color: #13161b;
    border: 1px solid #1e2229;
    border-radius: 10px;
    padding: 18px 20px 14px 20px;
    margin-bottom: 14px;
}
.setting-card-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a9eff;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e2229;
}
.pipeline-step {
    display: flex; align-items: center; gap: 8px;
    padding: 4px 0; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; color: #8a95a3;
}
.step-on  { color: #4ade80; }
.step-off { color: #f87171; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)
