import os
import glob
import uuid
from typing import List

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from openai import OpenAI

load_dotenv()

COLLECTION_NAME = "dsa_docs"
EMBED_DIM = 1536


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
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)
    openai = OpenAI()

    # Ensure collection (no drop/recreate to avoid churn)
    try:
        exists = client.collection_exists(collection_name=COLLECTION_NAME)
    except Exception:
        exists = False

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
        )

    docs = read_texts(data_dir)
    if not docs:
        print("No data found to ingest.")
        return

    # Stream embeddings and upserts per batch to keep memory low
    BATCH = 4
    total_chunks = 0
    batch_payloads: List[dict] = []

    def flush_batch(payloads_batch: List[dict]) -> int:
        if not payloads_batch:
            return 0
        texts = [p["text"] for p in payloads_batch]
        emb = openai.embeddings.create(model="text-embedding-3-small", input=texts)
        vecs = [d.embedding for d in emb.data]
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

    # Flush remaining
    total_chunks += flush_batch(batch_payloads)
    print(f"Ingested {total_chunks} chunks into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
