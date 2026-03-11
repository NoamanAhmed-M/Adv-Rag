import argparse
from pymilvus import connections, Collection, utility
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIARerank
from embedding_functions import get_embedding_function, llm
from data_real import relevancy_check
from gen_ans import generate_res, rewrite_query
from halluciation_check import hallucination_check
from final import final
from config import COLLECTION_NAME, get_milvus_connection_params, NVIDIA_API_KEY
import os

os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY

TOP_K           = 10    # retrieve more for reranker to work with
RERANK_TOP_N    = 5     # keep top 5 after reranking
SCORE_THRESHOLD = 0.25  # drop anything below this before reranking

# ── Reranker ──────────────────────────────────────────────────────────────────
reranker = NVIDIARerank(
    model="nvidia/nv-rerankqa-mistral-4b-v3",
    api_key=NVIDIA_API_KEY,
    top_n=RERANK_TOP_N,
    truncate="END",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)


def search_milvus(query_text: str, top_k: int = TOP_K):
    params = get_milvus_connection_params()
    print(f"[Milvus] Connecting to Zilliz Cloud...")
    connections.connect(**params)

    if not utility.has_collection(COLLECTION_NAME):
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' does not exist on Zilliz Cloud.\n"
            "Please upload documents and click 'Add to Database' first."
        )

    collection = Collection(COLLECTION_NAME)
    collection.load()

    embedding_function = get_embedding_function()
    query_embedding    = embedding_function.embed_query(query_text)

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        output_fields=["id", "text", "source", "page"],
    )

    docs_with_scores = []
    for hit in results[0]:
        doc = Document(
            page_content=hit.entity.get("text", ""),
            metadata={
                "id":     hit.entity.get("id", ""),
                "source": hit.entity.get("source", ""),
                "page":   hit.entity.get("page", 0),
            },
        )
        docs_with_scores.append((doc, hit.score))

    return docs_with_scores


def rerank_docs(query: str, docs_with_scores: list) -> list:
    """Rerank using nvidia/nv-rerankqa-mistral-4b-v3, fall back to original order."""
    try:
        docs = [doc for doc, _ in docs_with_scores]
        reranked = reranker.compress_documents(query=query, documents=docs)
        print(f"  [RERANK] {len(docs)} → {len(reranked)} docs after reranking")
        for i, doc in enumerate(reranked):
            score = doc.metadata.get("relevance_score", "N/A")
            print(f"  [RERANK] [{i+1}] relevance={score} | preview={doc.page_content[:80]}...")
        return reranked
    except Exception as e:
        print(f"  [RERANK] Failed ({e}) — using original order")
        return [doc for doc, _ in docs_with_scores]


def query_rag(query_text: str):

    print("\n" + "="*50)
    print(f"QUERY: {query_text}")
    print("="*50)

    # ── Step 1: Rewrite query ────────────────────────────────────────────────
    print("[STEP 1] Rewriting query...")
    search_query = rewrite_query(query_text)

    # ── Step 2: Search Milvus ────────────────────────────────────────────────
    print("[STEP 2] Searching Milvus...")
    results = search_milvus(search_query)
    print(f"[STEP 2] Returned {len(results)} results")
    for i, (doc, score) in enumerate(results):
        print(f"  [{i+1}] score={score:.4f} | preview={doc.page_content[:80]}...")

    # ── Step 3: Score pre-filter ─────────────────────────────────────────────
    print(f"[STEP 3] Score pre-filter (threshold={SCORE_THRESHOLD})...")
    above_threshold = [(doc, score) for doc, score in results if score >= SCORE_THRESHOLD]
    print(f"[STEP 3] {len(above_threshold)}/{len(results)} docs above threshold")

    if not above_threshold:
        print("[STEP 3] No docs above threshold — query out of scope")
        no_data_response = llm.invoke(
            f'The user asked: "{query_text}". The knowledge base has no relevant info. '
            "Politely inform them and suggest uploading relevant documents."
        )
        return no_data_response, no_data_response, "not_applicable"

    # ── Step 4: Rerank ───────────────────────────────────────────────────────
    # Use original question for reranking — measures relevance to what user asked
    print("[STEP 4] Reranking with nvidia/nv-rerankqa-mistral-4b-v3...")
    reranked_docs = rerank_docs(query_text, above_threshold)

    # ── Step 5: LLM relevancy check ──────────────────────────────────────────
    # Use original question here too — not the rewritten query
    print("[STEP 5] Running relevancy check...")
    docs_with_scores_reranked = [(doc, 1.0) for doc in reranked_docs]
    docs_to_use = relevancy_check(query_text, docs_with_scores_reranked)
    print(f"[STEP 5] {len(docs_to_use)}/{len(reranked_docs)} docs passed")

    if not docs_to_use:
        print("[STEP 5] All docs filtered — no relevant content in knowledge base")
        no_data_response = llm.invoke(
            f'The user asked: "{query_text}". No relevant documents found. '
            "Politely inform the user and suggest rephrasing or uploading relevant documents."
        )
        return no_data_response, no_data_response, "not_applicable"

    for doc in docs_to_use:
        print(f"[STEP 5] {doc.metadata}")

    # ── Step 6: Generate answer ───────────────────────────────────────────────
    print("[STEP 6] Generating response...")
    response_llm = generate_res(docs_to_use, query_text)
    print(f"[STEP 6] Response: {response_llm[:150]}...")

    # ── Step 7: Hallucination check ───────────────────────────────────────────
    print("[STEP 7] Hallucination check...")
    hallucination_binary_score = hallucination_check(docs_to_use, response_llm)
    print(f"[STEP 7] Score: {hallucination_binary_score}")

    # ── Step 8: Citations ─────────────────────────────────────────────────────
    print("[STEP 8] Building citations...")
    final_answer = final(docs_to_use, query_text, response_llm)
    print(f"[STEP 8] Sources : {getattr(final_answer, 'source', 'N/A')}")
    print(f"[STEP 8] Segments: {[s[:80] for s in getattr(final_answer, 'segment', [])]}")

    return response_llm, final_answer, hallucination_binary_score


if __name__ == "__main__":
    main()