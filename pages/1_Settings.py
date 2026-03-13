import streamlit as st
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from theme import inject_css, init_settings, DEFAULTS
from config import ADMIN_PASSWORD

st.set_page_config(
    page_title="Settings · RAG",
    layout="wide",
)
inject_css()
init_settings()

DATA_PATH = "/tmp/rag_data"
os.makedirs(DATA_PATH, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# AUTH GATE — full-page login wall
# ═════════════════════════════════════════════════════════════════════════════
if not st.session_state.get("authenticated", False):
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_pad, col_form, col_pad2 = st.columns([1, 1.2, 1])
    with col_form:
        st.markdown(
            """
            <div style="background:#13161b; border:1px solid #1e2229; border-radius:12px;
                        padding:36px 32px 28px 32px; text-align:center;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
                            letter-spacing:0.18em; text-transform:uppercase;
                            color:#4a5260; margin-bottom:6px;">RAG System</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:1.4rem;
                            font-weight:500; color:#e8edf2; margin-bottom:4px;">⚙️ Settings</div>
                <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.82rem;
                            color:#4a5260; margin-bottom:28px;">
                    Admin access required
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        pwd = st.text_input(
            "Password",
            type="password",
            placeholder="Enter admin password…",
            label_visibility="collapsed",
        )
        if st.button("→  Login", type="primary", use_container_width=True):
            if pwd == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password — try again.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def get_indexed_files() -> list[str]:
    try:
        from data_preprocessing import connect_milvus
        from config import COLLECTION_NAME
        from pymilvus import Collection, utility

        connect_milvus()
        if not utility.has_collection(COLLECTION_NAME):
            return []
        collection = Collection(COLLECTION_NAME)
        collection.load()
        rows = collection.query(expr="id != ''", output_fields=["source"], limit=16384)
        seen, names = set(), []
        for r in rows:
            base = os.path.basename(r.get("source", ""))
            if base and base not in seen:
                seen.add(base)
                names.append(base)
        return sorted(names)
    except Exception as e:
        st.error(f"Could not list indexed files: {e}")
        return []


def get_indexed_files_with_counts() -> list[tuple[str, str, int]]:
    """Return list of (basename, full_source_path, chunk_count) tuples."""
    try:
        from data_preprocessing import connect_milvus
        from config import COLLECTION_NAME
        from pymilvus import Collection, utility

        connect_milvus()
        if not utility.has_collection(COLLECTION_NAME):
            return []
        collection = Collection(COLLECTION_NAME)
        collection.load()
        rows = collection.query(expr="id != ''", output_fields=["source"], limit=16384)

        counts: dict[str, int] = {}
        full_paths: dict[str, str] = {}
        for r in rows:
            src = r.get("source", "")
            base = os.path.basename(src)
            if base:
                counts[base]     = counts.get(base, 0) + 1
                full_paths[base] = src  
        return sorted((base, full_paths[base], counts[base]) for base in counts)
    except Exception as e:
        st.error(f"Could not list indexed files: {e}")
        return []


def delete_file_from_milvus(source_basename: str) -> int:
    """
    Delete all vectors whose 'source' field ends with source_basename.
    Handles two known Milvus expression issues:
      - Backslashes in Windows paths must be escaped as \\\\ inside the string literal
      - Long IN expressions are batched to stay under Milvus parser limits
    Returns the number of entities deleted.
    """
    from data_preprocessing import connect_milvus
    from config import COLLECTION_NAME
    from pymilvus import Collection, utility

    connect_milvus()
    if not utility.has_collection(COLLECTION_NAME):
        return 0

    collection = Collection(COLLECTION_NAME)
    collection.load()

    rows = collection.query(
        expr="id != ''",
        output_fields=["id", "source"],
        limit=16384,
    )
    ids_to_delete = [
        r["id"] for r in rows
        if os.path.basename(r.get("source", "")) == source_basename
    ]

    if not ids_to_delete:
        return 0

    def _escape(pk: str) -> str:
        """Escape backslashes so Milvus expression parser doesn't choke on Windows paths."""
        return pk.replace("\\", "\\\\")

    BATCH = 50
    total_deleted = 0
    for i in range(0, len(ids_to_delete), BATCH):
        batch = ids_to_delete[i : i + BATCH]
        id_list = ", ".join(f'"{_escape(pk)}"' for pk in batch)
        expr = f"id in [{id_list}]"
        collection.delete(expr=expr)
        total_deleted += len(batch)
        print(f"[DELETE] Batch {i // BATCH + 1}: removed {len(batch)} chunk(s)")

    collection.flush()
    print(f"[DELETE] Total removed: {total_deleted} chunk(s) for '{source_basename}'")
    return total_deleted


def process_uploaded_files(uploaded_files, progress_callback=None):
    from data_preprocessing import load_documents, add_to_milvus
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    saved_files = []
    for uf in uploaded_files:
        dest = os.path.join(DATA_PATH, uf.name)
        with open(dest, "wb") as f:
            f.write(uf.getbuffer())
        saved_files.append(uf.name)

    documents = load_documents(DATA_PATH)
    mode = st.session_state.get("chunking_mode", "standard")

    if mode == "proposition":
        from proposition_chunking import proposition_chunk_documents
        chunks = proposition_chunk_documents(
            documents,
            pre_chunk_size     = st.session_state.get("prop_pre_chunk_size",    1000),
            pre_chunk_overlap  = st.session_state.get("prop_pre_chunk_overlap", 100),
            min_quality_score  = st.session_state.get("prop_min_quality_score", 6),
            quality_check      = st.session_state.get("prop_quality_check",     True),
            progress_callback  = progress_callback,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size    = st.session_state.get("chunk_size", 800),
            chunk_overlap = st.session_state.get("chunk_overlap", 80),
            length_function=len,
            is_separator_regex=False,
        )
        chunks = splitter.split_documents(documents)

    add_to_milvus(chunks)
    return saved_files, len(chunks)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═════════════════════════════════════════════════════════════════════════════
header_col, logout_col = st.columns([5, 1])
with header_col:
    st.title("Settings & Document Management")
    st.caption("All pipeline changes apply immediately to the next query. Document operations persist in Zilliz Cloud.")
with logout_col:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# ── Live pipeline status bar ──────────────────────────────────────────────────
steps = [
    ("use_query_rewrite",       "Query Rewrite"),
    ("use_relevancy_check",     "Relevancy Check"),
    ("use_reranker",            "Reranker"),
    ("use_hallucination_check", "Hallucination Check"),
]
cols = st.columns(len(steps))
for col, (key, label) in zip(cols, steps):
    on     = st.session_state.get(key, True)
    colour = "#4ade80" if on else "#f87171"
    border = "#1e4d2b" if on else "#4d1e1e"
    icon   = "●" if on else "○"
    col.markdown(
        f"""<div style="text-align:center;padding:10px 0;background:#13161b;
            border:1px solid {border};border-radius:8px;">
            <span style="color:{colour};font-family:'IBM Plex Mono',monospace;
            font-size:0.72rem;letter-spacing:0.08em;">{icon} {label}</span>
            </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_docs, tab_pipeline, tab_retrieval, tab_generation = st.tabs([
    "Documents",
    "Pipeline",
    "Retrieval & Chunking",
    "Generation & Display",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DOCUMENTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_docs:
    col_up, col_db = st.columns(2, gap="large")

    # ── Upload ────────────────────────────────────────────────────────────────
    with col_up:
        st.markdown('<div class="setting-card"><div class="setting-card-title">Upload Documents</div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Drop PDF or image files here",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file(s) selected**")
            for uf in uploaded_files:
                size_kb = round(len(uf.getvalue()) / 1024, 1)
                st.markdown(
                    f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;'
                    f'color:#8a95a3;padding:2px 0;"> {uf.name} <span style="color:#4a5260">({size_kb} KB)</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            mode = st.session_state.get("chunking_mode", "standard")
            if mode == "proposition":
                st.info(
                    "**Proposition mode** — LLM will extract atomic facts from each document. "
                    "Indexing will be slow. Configure in the Retrieval & Chunking tab.",
                   
                )
            else:
                st.info(
                    f"**Standard mode** — chunk size **{st.session_state.get('chunk_size', 800)}** / "
                    f"overlap **{st.session_state.get('chunk_overlap', 80)}** chars. "
                    "Change in the Retrieval & Chunking tab.",

                )

            if st.button("⬆  Add to Database", type="primary", use_container_width=True):
                prog_bar  = st.progress(0, text="Starting…")
                prog_text = st.empty()

                def _progress(current, total, message):
                    pct = int((current / total) * 100) if total else 100
                    prog_bar.progress(pct, text=message)
                    prog_text.caption(f"Step {current}/{total} — {message}")

                try:
                    saved, n_chunks = process_uploaded_files(
                        uploaded_files,
                        progress_callback=_progress if mode == "proposition" else None,
                    )
                    for f in saved:
                        if f not in st.session_state.uploaded_file_names:
                            st.session_state.uploaded_file_names.append(f)
                    prog_bar.progress(100, text="Done")
                    st.success(f"{len(saved)} file(s) indexed — {n_chunks} chunk(s) stored.")
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.error(traceback.format_exc())
        else:
            st.markdown(
                '<div style="text-align:center;padding:28px 0;color:#4a5260;'
                'font-family:\'IBM Plex Mono\',monospace;font-size:0.80rem;">'
                'No files selected yet</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Database management ───────────────────────────────────────────────────
    with col_db:
        st.markdown('<div class="setting-card"><div class="setting-card-title">Database</div>', unsafe_allow_html=True)

        # Init per-file confirm state
        if "confirm_delete_file" not in st.session_state:
            st.session_state.confirm_delete_file = None

        # Files currently indexed
        with st.spinner("Loading indexed files…"):
            indexed = get_indexed_files_with_counts()

        if indexed:
            st.markdown(f"**{len(indexed)} file(s) in knowledge base**")
            st.markdown("")

            for base, full_src, chunk_count in indexed:
                row_l, row_r = st.columns([5, 1])

                with row_l:
                    st.markdown(
                        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.74rem;'
                        f'color:#8a95a3;padding:4px 0;line-height:1.5;">'
                        f'{base}'
                        f'<span style="color:#4a5260;font-size:0.68rem;margin-left:8px;">'
                        f'({chunk_count} chunk{"s" if chunk_count != 1 else ""})</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                with row_r:
                    # If this file is pending confirmation, show confirm/cancel
                    if st.session_state.confirm_delete_file == base:
                        pass  # handled below
                    else:
                        if st.button("🗑", key=f"del_{base}", help=f"Delete {base}"):
                            st.session_state.confirm_delete_file = base
                            st.rerun()

                # Confirmation row (full width, below the file row)
                if st.session_state.confirm_delete_file == base:
                    st.warning(f"Delete **{base}** ({chunk_count} chunks) from the database?")
                    ca, cb = st.columns(2)
                    with ca:
                        if st.button("✓ Confirm", type="primary",
                                     key=f"confirm_del_{base}", use_container_width=True):
                            with st.spinner(f"Deleting {base}…"):
                                try:
                                    n = delete_file_from_milvus(base)
                                    # Also remove local temp file if present
                                    local_path = os.path.join(DATA_PATH, base)
                                    if os.path.isfile(local_path):
                                        os.remove(local_path)
                                    if base in st.session_state.get("uploaded_file_names", []):
                                        st.session_state.uploaded_file_names.remove(base)
                                    st.session_state.confirm_delete_file = None
                                    st.success(f"✓ Deleted {n} chunk(s) for '{base}'")
                                    st.rerun()
                                except Exception as e:
                                    st.session_state.confirm_delete_file = None
                                    st.error(f"Error: {e}")
                                    st.error(traceback.format_exc())
                    with cb:
                        if st.button("✕ Cancel", key=f"cancel_del_{base}", use_container_width=True):
                            st.session_state.confirm_delete_file = None
                            st.rerun()

                st.markdown(
                    '<hr style="border-color:#1e2229;margin:4px 0;">',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="text-align:center;padding:20px 0;color:#4a5260;'
                'font-family:\'IBM Plex Mono\',monospace;font-size:0.80rem;">'
                'Knowledge base is empty</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

        # ── System status ─────────────────────────────────────────────────────
        from config import ZILLIZ_URI, ZILLIZ_TOKEN, OCR_BACKEND, LLM_MODEL, COLLECTION_NAME
        rows = [
            ("Zilliz Cloud",    "✓ Connected"       if (ZILLIZ_URI and ZILLIZ_TOKEN) else "✗ Not configured",
                                "#4ade80"           if (ZILLIZ_URI and ZILLIZ_TOKEN) else "#f87171"),
            ("Collection",      COLLECTION_NAME,     "#c8d0d9"),
            ("OCR Backend",     OCR_BACKEND,         "#c8d0d9"),
            ("LLM Model",       LLM_MODEL,           "#c8d0d9"),
        ]
        for label, value, colour in rows:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
                f'font-family:\'IBM Plex Mono\',monospace;font-size:0.73rem;">'
                f'<span style="color:#4a5260">{label}</span>'
                f'<span style="color:{colour}">{value}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Clear database ────────────────────────────────────────────────────
        if not st.session_state.get("confirm_clear", False):
            if st.button("🗑  Clear Entire Database", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.error("This will **permanently delete all vectors** from Zilliz Cloud. This cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✓  Confirm Delete", type="primary", use_container_width=True):
                    with st.spinner("Clearing database…"):
                        try:
                            import data_preprocessing as dp
                            dp.clear_database()
                            if os.path.exists(DATA_PATH):
                                for f in os.listdir(DATA_PATH):
                                    fp = os.path.join(DATA_PATH, f)
                                    if os.path.isfile(fp):
                                        os.remove(fp)
                            st.session_state.confirm_clear   = False
                            st.session_state.uploaded_file_names = []
                            st.success("✓ Database cleared.")
                            st.balloons()
                            st.rerun()
                        except Exception as e:
                            st.session_state.confirm_clear = False
                            st.error(f"Error: {e}")
                            st.error(traceback.format_exc())
            with c2:
                if st.button("✕  Cancel", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
with tab_pipeline:
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="setting-card"><div class="setting-card-title">Pipeline Steps</div>', unsafe_allow_html=True)

        st.session_state.use_query_rewrite = st.toggle(
            "Query Rewrite",
            value=st.session_state.use_query_rewrite,
            help="Rewrite the user question into a dense keyword-rich search query before hitting Milvus.",
        )
        st.caption("Uses an extra LLM call to expand vague questions into better search queries. Helps recall.")

        st.markdown("---")

        st.session_state.use_relevancy_check = st.toggle(
            "Relevancy Check",
            value=st.session_state.use_relevancy_check,
            help="LLM grades each retrieved chunk — removes off-topic results before generation.",
        )
        st.caption("Adds ~1 LLM call per chunk. Filters noise but increases latency.")

        st.markdown("---")

        st.session_state.use_reranker = st.toggle(
            "Reranker  (nvidia/nv-rerankqa-mistral-4b-v3)",
            value=st.session_state.use_reranker,
            help="Cross-encoder reranking after initial vector search — better precision.",
        )
        st.caption("Nvidia cross-encoder reranker. Significantly improves result ordering. Recommended on.")

        st.markdown("---")

        st.session_state.use_hallucination_check = st.toggle(
            "Hallucination Check",
            value=st.session_state.use_hallucination_check,
            help="Checks whether the generated answer is grounded in the retrieved documents.",
        )
        st.caption("Adds a grounding badge to each answer. One extra LLM call per query.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="setting-card"><div class="setting-card-title">Display Options</div>', unsafe_allow_html=True)

        st.session_state.show_raw_sources = st.toggle(
            "Show source citations panel",
            value=st.session_state.show_raw_sources,
            help="Show the expandable sources panel below each answer in the chat.",
        )

        st.session_state.show_hallucination_badge = st.toggle(
            "Show hallucination badge",
            value=st.session_state.show_hallucination_badge,
            help="Display the Grounded / May not be grounded badge inside the sources panel.",
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Pipeline cost summary ─────────────────────────────────────────────
        st.markdown('<div class="setting-card"><div class="setting-card-title">Estimated LLM Calls / Query</div>', unsafe_allow_html=True)

        base = 1  
        extra = 0
        details = [("Answer generation", 1)]
        if st.session_state.use_query_rewrite:
            extra += 1
            details.append(("Query rewrite", 1))
        if st.session_state.use_relevancy_check:
            n = st.session_state.get("rerank_top_n", 5)
            extra += n
            details.append((f"Relevancy check (×{n} chunks)", n))
        if st.session_state.use_hallucination_check:
            extra += 1
            details.append(("Hallucination check", 1))

        total = base + extra
        for label, count in details:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:3px 0;'
                f'font-family:\'IBM Plex Mono\',monospace;font-size:0.73rem;">'
                f'<span style="color:#8a95a3">{label}</span>'
                f'<span style="color:#c8d0d9">+{count}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:6px 0 2px 0;'
            f'border-top:1px solid #1e2229;margin-top:6px;'
            f'font-family:\'IBM Plex Mono\',monospace;font-size:0.80rem;">'
            f'<span style="color:#4a9eff;font-weight:600">Total</span>'
            f'<span style="color:#4a9eff;font-weight:600">{total} calls</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — RETRIEVAL & CHUNKING
# ─────────────────────────────────────────────────────────────────────────────
with tab_retrieval:
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="setting-card"><div class="setting-card-title">Retrieval Parameters</div>', unsafe_allow_html=True)

        st.session_state.top_k = st.slider(
            "Top-K — chunks fetched from Milvus",
            min_value=1, max_value=50,
            value=st.session_state.top_k, step=1,
            help="How many candidate chunks to retrieve from the vector database before reranking.",
        )
        st.caption(f"**{st.session_state.top_k}** chunks retrieved from Milvus per query.")

        st.markdown("")

        st.session_state.rerank_top_n = st.slider(
            "Rerank Top-N — kept after reranking",
            min_value=1, max_value=st.session_state.top_k,
            value=min(st.session_state.rerank_top_n, st.session_state.top_k), step=1,
            help="How many chunks survive reranking and are passed to the LLM. Must be ≤ Top-K.",
        )
        st.caption(f"**{st.session_state.rerank_top_n}** chunks passed to answer generation.")

        st.markdown("")

        st.session_state.score_threshold = st.slider(
            "Cosine Score Threshold",
            min_value=0.0, max_value=1.0,
            value=float(st.session_state.score_threshold),
            step=0.01, format="%.2f",
            help="Chunks below this cosine similarity score are dropped before reranking.",
        )
        st.caption(
            f"**{st.session_state.score_threshold:.2f}** — "
            + ("strict filter" if st.session_state.score_threshold >= 0.5
               else "wide net — more chunks pass through" if st.session_state.score_threshold <= 0.15
               else "balanced")
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="setting-card"><div class="setting-card-title">Chunking Mode</div>', unsafe_allow_html=True)

        st.info(
            "Chunking settings only apply when uploading **new** documents. "
            "Already-indexed chunks are not re-split.",
        )

        mode = st.radio(
            "Select chunking strategy",
            options=["standard", "proposition"],
            format_func=lambda x: " Standard (character split)" if x == "standard" else " Proposition (LLM-based atomic facts)",
            index=0 if st.session_state.get("chunking_mode", "standard") == "standard" else 1,
            horizontal=True,
        )
        st.session_state.chunking_mode = mode

        st.markdown("---")

        if mode == "standard":
            st.session_state.chunk_size = st.slider(
                "Chunk Size (characters)",
                min_value=100, max_value=4000,
                value=st.session_state.chunk_size, step=50,
                help="Maximum number of characters per text chunk.",
            )
            st.session_state.chunk_overlap = st.slider(
                "Chunk Overlap (characters)",
                min_value=0, max_value=st.session_state.chunk_size // 2,
                value=min(st.session_state.chunk_overlap, st.session_state.chunk_size // 2),
                step=10,
                help="Characters shared between consecutive chunks.",
            )
            st.markdown(
                f'<div style="background:#0a0c0f;border:1px solid #1e2229;border-radius:6px;'
                f'padding:12px 14px;font-family:\'IBM Plex Mono\',monospace;font-size:0.73rem;">'
                f'<span style="color:#4a5260">chunk_size    </span>'
                f'<span style="color:#7dd3fc">{st.session_state.chunk_size}</span><br>'
                f'<span style="color:#4a5260">chunk_overlap  </span>'
                f'<span style="color:#7dd3fc">{st.session_state.chunk_overlap}</span><br>'
                f'<span style="color:#4a5260">effective_step </span>'
                f'<span style="color:#7dd3fc">{st.session_state.chunk_size - st.session_state.chunk_overlap}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        else:  # proposition mode
            st.markdown(
                '''<div style="background:rgba(74,158,255,0.06);border:1px solid rgba(74,158,255,0.15);
                border-radius:8px;padding:12px 14px;font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;
                color:#8a95a3;line-height:1.7;">
                <b style="color:#4a9eff">How it works:</b><br>
                1️ &nbsp;Documents are pre-split into intermediate chunks<br>
                2️ &nbsp;LLM decomposes each chunk into atomic factual statements<br>
                3️ &nbsp;Each proposition is graded on accuracy, clarity, completeness, conciseness<br>
                4️ &nbsp;Passing propositions are indexed — one vector per atomic fact
                </div>''',
                unsafe_allow_html=True,
            )
            st.markdown("")

            st.session_state.prop_pre_chunk_size = st.slider(
                "Pre-chunk Size (characters)",
                min_value=200, max_value=4000,
                value=st.session_state.get("prop_pre_chunk_size", 1000), step=50,
                help="Size of intermediate chunks fed to the LLM extractor. Larger = more context per LLM call.",
            )
            st.session_state.prop_pre_chunk_overlap = st.slider(
                "Pre-chunk Overlap (characters)",
                min_value=0, max_value=st.session_state.prop_pre_chunk_size // 2,
                value=min(st.session_state.get("prop_pre_chunk_overlap", 100),
                          st.session_state.prop_pre_chunk_size // 2),
                step=25,
                help="Overlap between intermediate chunks so propositions at boundaries aren't lost.",
            )
            st.markdown("---")
            st.session_state.prop_quality_check = st.toggle(
                "Enable quality grading",
                value=st.session_state.get("prop_quality_check", True),
                help="Run a second LLM call to score each proposition and drop low-quality ones.",
            )
            if st.session_state.prop_quality_check:
                st.session_state.prop_min_quality_score = st.slider(
                    "Minimum quality score (0–10, all axes)",
                    min_value=0, max_value=10,
                    value=st.session_state.get("prop_min_quality_score", 6),
                    step=1,
                    help="A proposition must score at least this on accuracy, clarity, completeness AND conciseness to be kept.",
                )
                score = st.session_state.prop_min_quality_score
                if score <= 3:
                    st.caption("Lenient — most propositions pass.")
                elif score <= 6:
                    st.caption("Balanced — moderate filtering.")
                else:
                    st.caption("Strict — only high-quality propositions are indexed.")

            st.markdown("")
            st.warning(
                "Proposition chunking makes **2 LLM calls per intermediate chunk** "
                "(1 extraction + 1 grading). Indexing will be significantly slower than standard chunking.",
            )

        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — GENERATION & DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
with tab_generation:
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="setting-card"><div class="setting-card-title">Generation Parameters</div>', unsafe_allow_html=True)

        st.session_state.llm_temperature = st.slider(
            "LLM Temperature",
            min_value=0.0, max_value=1.0,
            value=float(st.session_state.llm_temperature),
            step=0.05, format="%.2f",
            help="0 = deterministic / focused. Higher = more varied and creative.",
        )
        if st.session_state.llm_temperature == 0.0:
            st.caption("**Deterministic** — best for factual document Q&A.")
        elif st.session_state.llm_temperature <= 0.3:
            st.caption("**Low variance** — slightly more natural phrasing.")
        elif st.session_state.llm_temperature <= 0.7:
            st.caption("**Balanced** — may paraphrase differently between runs.")
        else:
            st.caption("**Creative** — answers can vary significantly. Not ideal for RAG.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="setting-card"><div class="setting-card-title">Chat Display</div>', unsafe_allow_html=True)

        st.session_state.show_raw_sources = st.toggle(
            "Show source citations panel",
            value=st.session_state.show_raw_sources,
        )
        st.caption("Expandable panel below each answer showing which document chunks were used.")

        st.markdown("")

        st.session_state.show_hallucination_badge = st.toggle(
            "Show hallucination grounding badge",
            value=st.session_state.show_hallucination_badge,
        )
        st.caption("Shows ✓ Grounded / ⚠ May not be grounded inside the citations panel.")

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🗑  Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat history cleared.")


# ═════════════════════════════════════════════════════════════════════════════
# BOTTOM BAR — reset / export / summary
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
bc1, bc2, bc3 = st.columns([1, 1, 3])

with bc1:
    if st.button("↺  Reset to Defaults", use_container_width=True):
        for key, val in DEFAULTS.items():
            st.session_state[key] = val
        st.success("Settings reset to defaults.")
        st.rerun()

with bc2:
    summary = "\n".join(f"{k}: {st.session_state.get(k, v)}" for k, v in DEFAULTS.items())
    st.download_button(
        "↓  Export Settings",
        data=summary,
        file_name="rag_settings.txt",
        mime="text/plain",
        use_container_width=True,
    )

with bc3:
    mode = st.session_state.get("chunking_mode", "standard")
    chunk_summary = (
        f"Prop (pre={st.session_state.get('prop_pre_chunk_size',1000)}, "
        f"min_score={st.session_state.get('prop_min_quality_score',6)})"
        if mode == "proposition"
        else f"{st.session_state.chunk_size}/{st.session_state.chunk_overlap} chars"
    )
    st.caption(
        f"Top-K **{st.session_state.top_k}**  ·  "
        f"Rerank-N **{st.session_state.rerank_top_n}**  ·  "
        f"Threshold **{st.session_state.score_threshold:.2f}**  ·  "
        f"Chunk **{chunk_summary}**  ·  "
        f"Mode **{mode}**  ·  "
        f"Temp **{st.session_state.llm_temperature:.2f}**"
    )

st.caption("Settings are stored in the Streamlit session — they reset when the browser tab is closed.")
