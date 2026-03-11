from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import base64
import httpx
from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility,
)
from pathlib import Path
from pdf2image import convert_from_path
from embedding_functions import get_embedding_function

# ── All config now comes from config.py / .env ────────────────────────────────
from config import (
    DATA_PATH, COLLECTION_NAME, EMBEDDING_DIM,
    OCR_BACKEND,
    NVIDIA_API_KEY, NVIDIA_API_URL, NVIDIA_OCR_MODEL,
    get_milvus_connection_params, print_config,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
PDF_EXTENSIONS   = {".pdf"}


# ══════════════════════════════════════════════════════════════════════════════
# OCR helpers
# ══════════════════════════════════════════════════════════════════════════════

def _read_image_b64(image_path: str) -> tuple[str, str]:
    ext = Path(image_path).suffix.lower().lstrip(".")
    media_type = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    print(f"  [b64] Reading {Path(image_path).name} as {media_type}")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    print(f"  [b64] Encoded size: {len(data) // 1024} KB")
    return data, media_type


def nvidia_ocr_image(image_path: str) -> str:
    """
    OCR via Nvidia-hosted NeMo Retriever OCR v1.
    Hosted endpoint : https://integrate.api.nvidia.com/v1/infer
    Payload format  : { "input": [{ "type": "image_url", "url": "data:<mime>;base64,<b64>" }],
                        "merge_levels": ["paragraph"] }
    Response format : { "data": [{ "text_detections": [{ "text_prediction": { "text": "..." } }] }] }
    Docs: https://docs.nvidia.com/nim/ingestion/image-ocr/latest/api-reference.html
    """
    endpoint = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr"
    print(f"  [Nvidia OCR] {Path(image_path).name} -> {endpoint}")

    b64_data, media_type = _read_image_b64(image_path)

    # Only png and jpeg are supported — convert tiff/bmp at call site if needed
    payload = {
        "input": [
            {
                "type": "image_url",
                "url":  f"data:{media_type};base64,{b64_data}",
            }
        ],
        "merge_levels": ["paragraph"],
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }

    response = httpx.post(endpoint, json=payload, headers=headers, timeout=120)
    print(f"  [Nvidia OCR] Response status: {response.status_code}")
    response.raise_for_status()

    # Parse response: data[].text_detections[].text_prediction.text
    result = response.json()
    texts  = []
    for page in result.get("data", []):
        for det in page.get("text_detections", []):
            t = det.get("text_prediction", {}).get("text", "").strip()
            if t:
                texts.append(t)
    text = "\n".join(texts)
    print(f"  [Nvidia OCR] Extracted {len(text)} chars from {len(texts)} detection(s)")
    return text


def ocr_image(image_path: str) -> str:
    """Run OCR on an image file. Uses Nvidia NIM only — ollama not supported on cloud."""
    print(f"[OCR] Processing: {Path(image_path).name} (backend={OCR_BACKEND})")

    if OCR_BACKEND not in ("nvidia", "both"):
        raise ValueError(
            f"OCR_BACKEND='{OCR_BACKEND}' is not supported on cloud. "
            "Set OCR_BACKEND=nvidia in your .env file."
        )

    try:
        text = nvidia_ocr_image(image_path)
        print(f"  [OCR] Nvidia succeeded for {Path(image_path).name}")
        return text
    except Exception as nvidia_err:
        print(f"  [OCR] Nvidia failed: {nvidia_err}")
        raise


# ══════════════════════════════════════════════════════════════════════════════
# Milvus helpers
# ══════════════════════════════════════════════════════════════════════════════

def connect_milvus():
    params = get_milvus_connection_params()
    print(f"[Milvus] Connecting to Zilliz Cloud...")
    connections.connect(**params)
    print("[Milvus] Connected.")


def get_or_create_collection() -> Collection:
    if utility.has_collection(COLLECTION_NAME):
        print(f"[Milvus] Collection '{COLLECTION_NAME}' already exists — reusing.")
        return Collection(COLLECTION_NAME)

    print(f"[Milvus] Creating new collection '{COLLECTION_NAME}' (dim={EMBEDDING_DIM})...")
    fields = [
        FieldSchema(name="id",        dtype=DataType.VARCHAR,      is_primary=True, max_length=512),
        FieldSchema(name="text",      dtype=DataType.VARCHAR,      max_length=65535),
        FieldSchema(name="source",    dtype=DataType.VARCHAR,      max_length=512),
        FieldSchema(name="page",      dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    schema     = CollectionSchema(fields, description="RAG document store")
    collection = Collection(COLLECTION_NAME, schema)
    collection.create_index(
        field_name="embedding",
        index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
    )
    print(f"[Milvus] Collection '{COLLECTION_NAME}' created with IVF_FLAT COSINE index.")
    return collection


# ══════════════════════════════════════════════════════════════════════════════
# Document loading
# ══════════════════════════════════════════════════════════════════════════════

def load_documents(data_path: str) -> list[Document]:
    print(f"\n[Load] Scanning '{data_path}' for documents...")
    documents = []
    all_files = list(Path(data_path).glob("*"))
    print(f"[Load] Found {len(all_files)} file(s) in directory.")

    for file_path in all_files:
        ext = file_path.suffix.lower()
        print(f"\n[Load] Processing: {file_path.name} (ext={ext})")

        if ext in PDF_EXTENSIONS:
            print(f"  [PDF] Loading with PyPDFLoader...")
            loader = PyPDFLoader(str(file_path))
            docs   = loader.load()
            print(f"  [PDF] Loaded {len(docs)} page(s).")

            if not any(doc.page_content.strip() for doc in docs):
                print(f"  [PDF] No text found — triggering OCR fallback for scanned PDF.")
                images = convert_from_path(str(file_path))
                print(f"  [PDF] Converted to {len(images)} image(s) for OCR.")
                for i, img in enumerate(images):
                    temp_path = f"temp_page_{i}.png"
                    img.save(temp_path)
                    print(f"  [PDF] OCR on page {i + 1}/{len(images)}...")
                    text = ocr_image(temp_path)
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": str(file_path), "page": i},
                    ))
                    os.remove(temp_path)
            else:
                total_chars = sum(len(d.page_content) for d in docs)
                print(f"  [PDF] Text extracted — {total_chars} total chars across {len(docs)} page(s).")
                documents.extend(docs)

        elif ext in IMAGE_EXTENSIONS:
            print(f"  [Image] Running OCR...")
            text = ocr_image(str(file_path))
            documents.append(Document(
                page_content=text,
                metadata={"source": str(file_path), "page": 0},
            ))
        else:
            print(f"  [Skip] Unsupported file type.")

    print(f"\n[Load] Done. Total documents loaded: {len(documents)}")
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# Splitting & ID assignment
# ══════════════════════════════════════════════════════════════════════════════

def split_document(documents: list[Document]) -> list[Document]:
    print(f"\n[Split] Splitting {len(documents)} document(s) into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80,
        length_function=len, is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    print(f"[Split] Produced {len(chunks)} chunk(s).")
    return chunks


def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    print(f"[IDs] Assigning chunk IDs to {len(chunks)} chunk(s)...")
    last_page_id, current_chunk_index = None, 0
    for chunk in chunks:
        page_id = f"{chunk.metadata.get('source')}:{chunk.metadata.get('page')}"
        current_chunk_index = (current_chunk_index + 1) if page_id == last_page_id else 0
        chunk.metadata["id"] = f"{page_id}:{current_chunk_index}"
        last_page_id = page_id
    print(f"[IDs] Done.")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Milvus insert with dedup
# ══════════════════════════════════════════════════════════════════════════════

def add_to_milvus(chunks: list[Document], similarity_threshold=0.999, batch_size=25):
    print(f"\n[Insert] Starting Milvus insert for {len(chunks)} chunk(s)...")
    connect_milvus()
    collection = get_or_create_collection()
    print("[Milvus] Loading collection into memory...")
    collection.load()
    print("[Milvus] Collection loaded.")

    chunks_with_ids    = calculate_chunk_ids(chunks)
    print("[Embed] Initialising embedding function...")
    embedding_function = get_embedding_function()
    print("[Embed] Embedding function ready.")

    print("[Dedup] Fetching existing IDs from Milvus...")
    existing_ids: set[str] = set()
    offset, page_size = 0, 1000
    while True:
        rows = collection.query(expr="id != ''", output_fields=["id"],
                                offset=offset, limit=page_size)
        if not rows:
            break
        existing_ids.update(r["id"] for r in rows)
        offset += len(rows)
        if len(rows) < page_size:
            break

    print(f"[Dedup] Existing documents in Milvus: {len(existing_ids)}")

    id_filtered = [c for c in chunks_with_ids if c.metadata["id"] not in existing_ids]
    skipped_by_id = len(chunks_with_ids) - len(id_filtered)
    print(f"[Dedup] Skipped {skipped_by_id} already-known chunk(s) by ID.")
    print(f"[Dedup] {len(id_filtered)} chunk(s) remain after ID filtering.")
    if not id_filtered:
        print("[Insert] No new documents to add.")
        return

    new_chunks, new_embeddings = [], []
    total_batches = (len(id_filtered) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(id_filtered), batch_size), start=1):
        batch = id_filtered[i : i + batch_size]
        print(f"\n[Embed] Batch {batch_num}/{total_batches} — embedding {len(batch)} chunk(s)...")
        batch_embeddings = embedding_function.embed_documents([c.page_content for c in batch])
        print(f"[Embed] Batch {batch_num} embeddings done.")

        skipped_similar = 0
        for chunk, emb in zip(batch, batch_embeddings):
            if existing_ids or new_chunks:
                results = collection.search(
                    data=[emb], anns_field="embedding",
                    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                    limit=1, output_fields=["id"],
                )
                if results and results[0] and results[0][0].score >= similarity_threshold:
                    print(f"  [Dedup] Too similar (score {results[0][0].score:.4f}) — skipping: {chunk.metadata['id']}")
                    skipped_similar += 1
                    continue

            new_chunks.append(chunk)
            new_embeddings.append(emb)

        if skipped_similar:
            print(f"  [Dedup] Skipped {skipped_similar} chunk(s) by similarity in batch {batch_num}.")

        if len(new_chunks) >= batch_size:
            print(f"[Insert] Flushing sub-batch of {len(new_chunks)} chunk(s) to Milvus...")
            collection.insert([
                [c.metadata["id"]               for c in new_chunks],
                [c.page_content[:65535]         for c in new_chunks],
                [c.metadata.get("source", "")   for c in new_chunks],
                [int(c.metadata.get("page", 0)) for c in new_chunks],
                new_embeddings,
            ])
            collection.flush()
            print(f"[Insert] Sub-batch flushed successfully.")
            new_chunks.clear()
            new_embeddings.clear()

    if new_chunks:
        print(f"\n[Insert] Inserting final batch of {len(new_chunks)} chunk(s)...")
        collection.insert([
            [c.metadata["id"]               for c in new_chunks],
            [c.page_content[:65535]         for c in new_chunks],
            [c.metadata.get("source", "")   for c in new_chunks],
            [int(c.metadata.get("page", 0)) for c in new_chunks],
            new_embeddings,
        ])
        collection.flush()
        print("[Insert] Final batch flushed successfully.")
    else:
        print("[Insert] No remaining chunks to flush.")

    print("\n[Insert] Insert complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Clear database
# ══════════════════════════════════════════════════════════════════════════════

def clear_database():
    print(f"[Clear] Connecting to Milvus to drop collection '{COLLECTION_NAME}'...")
    connect_milvus()
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"[Clear] Collection '{COLLECTION_NAME}' dropped.")
    else:
        print(f"[Clear] Collection '{COLLECTION_NAME}' does not exist — nothing to drop.")


def main():
    print_config()
    print("=" * 60)
    print("data_preprocessing.py — starting")
    print("=" * 60)
    documents = load_documents(DATA_PATH)
    chunks    = split_document(documents)
    add_to_milvus(chunks)
    print("\n[Done] Pipeline complete.")


if __name__ == "__main__":
    main()
