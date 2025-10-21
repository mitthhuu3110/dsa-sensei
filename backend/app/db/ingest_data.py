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


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> List[str]:
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

    # Ensure collection
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
        )
    except Exception:
        pass

    docs = read_texts(data_dir)
    payloads = []
    vectors = []
    ids = []

    for doc in docs:
        for chunk in chunk_text(doc["text"]):
            payloads.append({"text": chunk, "source": doc["source"]})

    if not payloads:
        print("No data found to ingest.")
        return

    # Embed in batches
    BATCH = 64
    for i in range(0, len(payloads), BATCH):
        batch = payloads[i:i+BATCH]
        texts = [p["text"] for p in batch]
        emb = openai.embeddings.create(model="text-embedding-3-small", input=texts)
        vectors.extend([d.embedding for d in emb.data])
        ids.extend([str(uuid.uuid4()) for _ in batch])

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=qmodels.Batch(ids=ids, vectors=vectors, payloads=payloads),
    )
    print(f"Ingested {len(payloads)} chunks into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
