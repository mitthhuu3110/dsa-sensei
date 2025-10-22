import os
import glob
import uuid
import time
from typing import List

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

load_dotenv()

COLLECTION_NAME = "dsa_docs"
EMBED_DIM_OPENAI = 1536
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()


def read_texts(data_dir: str) -> List[dict]:
    paths = glob.glob(os.path.join(data_dir, "**", "*"), recursive=True)
    texts = []
    for p in paths:
        if os.path.isdir(p):
            continue
        if any(p.endswith(ext) for ext in [".md", ".txt"]):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                texts.append({"text": f.read(), "source": os.path.relpath(p, data_dir)})
    return texts


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def main() -> None:
    # backend/app/db -> repo_root/data (go up 3 to backend, 1 more to repo root)
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../data"))
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)
    # Prepare embedder
    local_embedder = None
    if EMBEDDING_PROVIDER == "local":
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not available. Install it or set EMBEDDING_PROVIDER=openai")
        local_embedder = SentenceTransformer(LOCAL_EMBED_MODEL)

    # Ensure collection (no drop/recreate to avoid churn)
    try:
        exists = client.collection_exists(collection_name=COLLECTION_NAME)
    except Exception:
        exists = False

    if not exists:
        size = EMBED_DIM_OPENAI if EMBEDDING_PROVIDER == "openai" else 384
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=size, distance=qmodels.Distance.COSINE),
        )

    docs = read_texts(data_dir)
    if not docs:
        print("No data found to ingest.")
        return

    # Stream embeddings and upserts per batch to keep memory low
    BATCH = int(os.getenv("INGEST_BATCH", "1"))
    MAX_CHUNKS = int(os.getenv("INGEST_MAX_CHUNKS", "0"))  # 0 = no cap
    SLEEP_MS = int(os.getenv("INGEST_SLEEP_MS", "100"))
    total_chunks = 0
    batch_payloads: List[dict] = []

    def flush_batch(payloads_batch: List[dict]) -> int:
        if not payloads_batch:
            return 0
        texts = [p["text"] for p in payloads_batch]
        # Embeddings: local by default to avoid 429 and reduce latency
        if EMBEDDING_PROVIDER == "openai":
            from openai import OpenAI  # local import to avoid importing if unused
            openai_client = OpenAI()
            emb = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
            vecs = [d.embedding for d in emb.data]
        else:
            embs = local_embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            vecs = [e.tolist() for e in embs]
        ids = [str(uuid.uuid4()) for _ in payloads_batch]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=qmodels.Batch(ids=ids, vectors=vecs, payloads=payloads_batch),
        )
        return len(payloads_batch)

    for doc in docs:
        for chunk in chunk_text(doc["text"]):
            batch_payloads.append({"text": chunk, "source": doc["source"]})
            if len(batch_payloads) >= BATCH:
                total_chunks += flush_batch(batch_payloads)
                batch_payloads = []
                if SLEEP_MS > 0:
                    time.sleep(SLEEP_MS / 1000.0)
                if MAX_CHUNKS and total_chunks >= MAX_CHUNKS:
                    print(f"Stopping early at {total_chunks} chunks (MAX_CHUNKS)")
                    print(f"Ingested {total_chunks} chunks into '{COLLECTION_NAME}'.")
                    return

    # Flush remaining
    total_chunks += flush_batch(batch_payloads)
    print(f"Ingested {total_chunks} chunks into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
