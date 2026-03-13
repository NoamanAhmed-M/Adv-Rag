"""
Proposition Chunking
====================
Based on the "Dense X Retrieval" paper and NirDiamant's RAG Techniques notebook.

Pipeline per document chunk:
  1. LLM extracts atomic, self-contained factual propositions.
  2. A second LLM grades each proposition on 4 axes (accuracy, clarity,
     completeness, conciseness) and drops those that fall below threshold.
  3. Passing propositions become the chunks that get embedded into Milvus.

Each proposition is stored as a LangChain Document preserving the original
source / page metadata so citations still work end-to-end.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedding_functions import llm

# ── Prompts ───────────────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM = """You are an expert at decomposing text into atomic factual propositions.

Given a passage of text, extract every distinct factual claim as a separate proposition.

Rules for each proposition:
- It must be a single, complete, self-contained sentence.
- It must be understandable WITHOUT reading the surrounding text (no pronouns like
  "it", "they", "this" that refer to something outside the sentence).
- It must express exactly ONE fact or claim — do not bundle multiple facts.
- It must be faithful to the source text — do not invent or infer.
- Write in plain, clear English.

Respond with ONLY a JSON array of strings, no explanation, no markdown fences.
Example: ["The Eiffel Tower is located in Paris.", "It was built in 1889."]"""

_EXTRACTION_HUMAN = "Passage:\n\n{chunk}"

_GRADING_SYSTEM = """You are a strict quality grader for factual propositions.

Grade the proposition on these four axes, each from 0 to 10:
  - accuracy    : Is the proposition factually correct and faithful to the source?
  - clarity     : Is it written in clear, unambiguous language?
  - completeness: Does it contain all information needed to stand alone?
  - conciseness : Is it free of unnecessary words?

You MUST respond with ONLY a raw JSON object — no explanation, no markdown.
Example: {{"accuracy": 9, "clarity": 8, "completeness": 7, "conciseness": 9}}"""

_GRADING_HUMAN = (
    "Source passage:\n\n{chunk}\n\n"
    "Proposition to grade:\n\n{proposition}"
)

# ── Chains ────────────────────────────────────────────────────────────────────

_extraction_chain = (
    ChatPromptTemplate.from_messages(
        [("system", _EXTRACTION_SYSTEM), ("human", _EXTRACTION_HUMAN)]
    )
    | llm
    | StrOutputParser()
)

_grading_chain = (
    ChatPromptTemplate.from_messages(
        [("system", _GRADING_SYSTEM), ("human", _GRADING_HUMAN)]
    )
    | llm
    | StrOutputParser()
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_list(raw: str) -> list[str]:
    """Robustly extract a JSON array of strings from LLM output."""
    raw = raw.strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [str(s) for s in result]
    except Exception:
        pass
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return [str(s) for s in result]
    except Exception:
        pass
    # Last resort: find first [...] block
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    # Fallback: split by newlines, strip bullets
    lines = [re.sub(r'^[\s\-\*\d\.]+', '', l).strip() for l in raw.splitlines()]
    return [l for l in lines if len(l) > 10]


def _parse_json_obj(raw: str) -> dict:
    """Robustly extract a JSON object from LLM output."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in: {raw!r}")


# ── Core functions ────────────────────────────────────────────────────────────

def extract_propositions(chunk_text: str) -> list[str]:
    """Call the LLM to decompose a chunk into atomic propositions."""
    try:
        raw = _extraction_chain.invoke({"chunk": chunk_text})
        propositions = _parse_json_list(raw)
        # Filter out empty / very short strings
        return [p.strip() for p in propositions if len(p.strip()) > 15]
    except Exception as e:
        print(f"  [PROP] Extraction error: {e} — returning empty list")
        return []


def grade_proposition(
    proposition: str,
    source_chunk: str,
    min_score: int = 6,
) -> tuple[bool, dict]:
    """
    Grade a proposition on 4 axes.
    Returns (passed: bool, scores: dict).
    A proposition passes if ALL four scores >= min_score.
    """
    try:
        raw = _grading_chain.invoke(
            {"chunk": source_chunk, "proposition": proposition}
        )
        scores = _parse_json_obj(raw)
        axes = ["accuracy", "clarity", "completeness", "conciseness"]
        # Normalise keys to lowercase
        scores = {k.lower(): int(v) for k, v in scores.items() if k.lower() in axes}
        passed = all(scores.get(ax, 0) >= min_score for ax in axes)
        return passed, scores
    except Exception as e:
        print(f"  [PROP] Grading error: {e} — defaulting to pass")
        return True, {}


# ── Main entry point ──────────────────────────────────────────────────────────

def proposition_chunk_documents(
    documents: list[Document],
    pre_chunk_size: int = 1000,
    pre_chunk_overlap: int = 100,
    min_quality_score: int = 6,
    quality_check: bool = True,
    progress_callback=None,
) -> list[Document]:
    """
    Full proposition chunking pipeline.

    Args:
        documents:          Raw LangChain Documents (from load_documents).
        pre_chunk_size:     Size of intermediate chunks fed to the extractor LLM.
        pre_chunk_overlap:  Overlap for intermediate chunks.
        min_quality_score:  Minimum score (0-10) on each axis to keep a proposition.
        quality_check:      Whether to run the grading step at all.
        progress_callback:  Optional callable(current, total, message) for UI updates.

    Returns:
        List of Documents where each page_content is one passing proposition.
    """
    print(f"\n[PROP] Starting proposition chunking for {len(documents)} document(s)")

    # Step 1 — pre-split into manageable chunks for the LLM
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=pre_chunk_size,
        chunk_overlap=pre_chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    pre_chunks = splitter.split_documents(documents)
    print(f"[PROP] Pre-split into {len(pre_chunks)} intermediate chunk(s)")

    proposition_docs: list[Document] = []
    total = len(pre_chunks)

    for idx, chunk in enumerate(pre_chunks):
        msg = f"Chunk {idx + 1}/{total} — extracting propositions…"
        print(f"[PROP] {msg}")
        if progress_callback:
            progress_callback(idx, total, msg)

        # Step 2 — extract propositions
        propositions = extract_propositions(chunk.page_content)
        print(f"  [PROP] Extracted {len(propositions)} proposition(s)")

        for prop_idx, prop in enumerate(propositions):
            passed = True
            scores: dict = {}

            # Step 3 — quality gate
            if quality_check:
                passed, scores = grade_proposition(
                    prop, chunk.page_content, min_score=min_quality_score
                )
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  [PROP] [{prop_idx + 1}] {status} scores={scores} | {prop[:60]}…")
            else:
                print(f"  [PROP] [{prop_idx + 1}] (no grading) | {prop[:60]}…")

            if passed:
                # Build a new Document for this proposition, keeping source metadata
                prop_doc = Document(
                    page_content=prop,
                    metadata={
                        **chunk.metadata,
                        "chunk_method": "proposition",
                        "quality_scores": scores,
                        "source_chunk_preview": chunk.page_content[:200],
                    },
                )
                proposition_docs.append(prop_doc)

    if progress_callback:
        progress_callback(total, total, "Done")

    total_extracted = len(proposition_docs)
    print(
        f"\n[PROP] Done — {total_extracted} proposition(s) passed quality check "
        f"from {len(pre_chunks)} intermediate chunk(s)"
    )
    return proposition_docs
