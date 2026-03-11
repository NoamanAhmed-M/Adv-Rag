from pydantic import BaseModel, Field, model_validator
from typing import List

class HighlightDocuments(BaseModel):
    id: List[str] = Field(default_factory=list)
    source: List[str] = Field(default_factory=list)
    segment: List[str] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def normalize_keys(cls, values):
        return {k.lower(): v for k, v in values.items()}


def final(docs_to_use, question, generation):
    """
    Build citations by matching which retrieved chunks actually appear
    in the generated answer. Falls back to top-3 if no matches found.
    """
    try:
        matched_ids, matched_sources, matched_segments = [], [], []

        # Check which chunks have content referenced in the answer
        answer_lower = generation.lower()
        for doc in docs_to_use:
            # Take key phrases from the chunk (first 60 chars of each sentence)
            sentences = [s.strip() for s in doc.page_content.split('.') if len(s.strip()) > 20]
            matched = any(sent[:50].lower() in answer_lower for sent in sentences)

            if matched:
                matched_ids.append(doc.metadata.get("id", ""))
                matched_sources.append(doc.metadata.get("source", ""))
                matched_segments.append(doc.page_content[:500])

        # Fallback: if no phrase matches found, use top-3 by position (highest score first)
        if not matched_sources:
            print("  [FINAL] No phrase matches — using top-3 docs")
            for doc in docs_to_use[:3]:
                matched_ids.append(doc.metadata.get("id", ""))
                matched_sources.append(doc.metadata.get("source", ""))
                matched_segments.append(doc.page_content[:500])

        # Deduplicate by source
        seen, final_ids, final_sources, final_segments = set(), [], [], []
        for i, src in enumerate(matched_sources):
            if src not in seen:
                seen.add(src)
                final_ids.append(matched_ids[i])
                final_sources.append(src)
                final_segments.append(matched_segments[i])

        print(f"  [FINAL] {len(final_sources)} unique source(s) cited")
        return HighlightDocuments(id=final_ids, source=final_sources, segment=final_segments)

    except Exception as e:
        print(f"  [FINAL] ❌ Error: {e}")
        return HighlightDocuments(id=[], source=[], segment=[])