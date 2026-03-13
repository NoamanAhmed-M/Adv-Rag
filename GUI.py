import streamlit as st
import os
import traceback
import warnings

warnings.filterwarnings("ignore")
import logging
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from theme import inject_css, init_settings

st.set_page_config(
    page_title="RAG Chat",
    page_icon="💬",
    layout="wide",
)
inject_css()
init_settings()

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR  — status + pipeline summary only
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-section-header">System Status</div>', unsafe_allow_html=True)

    from config import ZILLIZ_URI, ZILLIZ_TOKEN, OCR_BACKEND
    if ZILLIZ_URI and ZILLIZ_TOKEN:
        st.markdown('<span class="status-badge connected">● Zilliz Cloud</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge disconnected">✕ Zilliz — credentials missing</span>', unsafe_allow_html=True)

    if st.session_state.get("authenticated", False):
        st.markdown('<span class="status-badge admin">● Admin</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge warning">○ Not logged in</span>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sidebar-section-header">Pipeline</div>', unsafe_allow_html=True)
    steps = [
        ("Query rewrite",       "use_query_rewrite"),
        ("Relevancy check",     "use_relevancy_check"),
        ("Reranker",            "use_reranker"),
        ("Hallucination check", "use_hallucination_check"),
    ]
    for label, key in steps:
        on  = st.session_state.get(key, True)
        dot = "🟢" if on else "🔴"
        st.markdown(
            f'<div class="pipeline-step">{dot} <span>{label}</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption(
        f"Top-K **{st.session_state.get('top_k', 10)}**  ·  "
        f"Rerank-N **{st.session_state.get('rerank_top_n', 5)}**  ·  "
        f"Threshold **{st.session_state.get('score_threshold', 0.25):.2f}**"
    )
    st.caption("Go to ⚙️ Settings to configure the pipeline.")
    st.divider()
    st.caption("Zilliz · Nvidia NIM · Nemotron · LangChain")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CHAT
# ═════════════════════════════════════════════════════════════════════════════
st.title("💬  RAG Document Chat")
st.caption("Retrieval-Augmented Generation over your documents · Configure & manage documents in ⚙️ Settings")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            hal      = message.get("hal_score")
            sources  = message.get("sources",  [])
            segments = message.get("segments", [])
            if sources and st.session_state.get("show_raw_sources", True):
                with st.expander(f"Sources ({len(sources)})"):
                    if st.session_state.get("show_hallucination_badge", True):
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

if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            try:
                from Query import query_rag
                answer_text, result, hal_score = query_rag(
                    query_text           = prompt,
                    top_k                = st.session_state.get("top_k", 10),
                    rerank_top_n         = st.session_state.get("rerank_top_n", 5),
                    score_threshold      = st.session_state.get("score_threshold", 0.25),
                    use_query_rewrite    = st.session_state.get("use_query_rewrite", True),
                    use_relevancy_check  = st.session_state.get("use_relevancy_check", True),
                    use_reranker         = st.session_state.get("use_reranker", True),
                    use_hallucination    = st.session_state.get("use_hallucination_check", True),
                )

                if isinstance(answer_text, str) and answer_text:
                    display_text = answer_text
                elif hasattr(result, "content"):
                    display_text = result.content
                elif isinstance(result, str):
                    display_text = result
                else:
                    display_text = str(result)

                st.markdown(display_text)

                score_val = (
                    hal_score.binary_score if hasattr(hal_score, "binary_score")
                    else hal_score         if isinstance(hal_score, str)
                    else "not_applicable"
                )

                sources, segments = [], []
                if hasattr(result, "source"):
                    sources  = result.source  if isinstance(result.source,  list) else [result.source]
                if hasattr(result, "segment"):
                    segments = result.segment if isinstance(result.segment, list) else [result.segment]

                if st.session_state.get("show_raw_sources", True):
                    with st.expander(f"Sources ({len(sources)})"):
                        if st.session_state.get("show_hallucination_badge", True):
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
                    "role":      "assistant",
                    "content":   display_text,
                    "sources":   sources,
                    "segments":  segments,
                    "hal_score": score_val,
                })

            except RuntimeError as e:
                msg = str(e)
                st.warning(msg, icon="📭")
                st.session_state.messages.append({"role": "assistant", "content": msg})

            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.error(traceback.format_exc())
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.divider()
st.caption("Built with Python · LangChain · Zilliz Cloud · Nvidia NIM")
