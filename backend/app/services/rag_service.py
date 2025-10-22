import os
import time
import uuid
from typing import Any, Dict, List

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from openai import OpenAI

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

load_dotenv()

COLLECTION_NAME = "dsa_docs"
EMBED_DIM_OPENAI = 1536  # text-embedding-3-small
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class RagService:
    def __init__(self) -> None:
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=self.qdrant_url)
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
        self.openai = OpenAI()
        self._ensure_collection()
        self._local_embedder = None

    def _ensure_collection(self) -> None:
        # Create if missing, do not drop existing collection
        try:
            if not self.client.collection_exists(collection_name=COLLECTION_NAME):
                # Choose vector size based on provider
                size = EMBED_DIM_OPENAI if self.embedding_provider == "openai" else 384
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=qmodels.VectorParams(size=size, distance=qmodels.Distance.COSINE),
                )
        except Exception as e:
            raise RuntimeError(f"Failed ensuring collection '{COLLECTION_NAME}': {e}")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if self.embedding_provider == "openai":
            response = self.openai.embeddings.create(model="text-embedding-3-small", input=texts)
            return [d.embedding for d in response.data]
        # local embeddings
        if self._local_embedder is None:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not available. Install it or set EMBEDDING_PROVIDER=openai")
            self._local_embedder = SentenceTransformer(LOCAL_EMBED_MODEL)
        embs = self._local_embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [e.tolist() for e in embs]

    def _search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        query_vec = self._embed([query])[0]
        res = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=k,
            with_payload=True,
        )
        docs: List[Dict[str, Any]] = []
        for p in res:
            payload = p.payload or {}
            docs.append({
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "score": float(p.score),
            })
        return docs

    def _compose_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        ctx_block = "\n\n".join([f"[Source: {c['source']}]\n{c['text']}" for c in contexts]) if contexts else ""
        system = (
            "You are DSA-Sensei, a friendly and motivational tutor.\n"
            "Explain this DSA concept clearly, step-by-step, with intuition and motivation.\n"
            "Use retrieved context as factual reference.\n"
            "Encourage the learner to stay consistent."
        )
        user = (
            f"Question: {question}\n\n"
            f"Retrieved context (may be partial, use prudently):\n{ctx_block}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _answer(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        # Try OpenAI if available and quota allows; otherwise fallback to a deterministic tutor response
        try:
            start = time.time()
            resp = self.openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
            )
            latency_ms = int((time.time() - start) * 1000)
            usage = getattr(resp, "usage", None)
            return {
                "answer": resp.choices[0].message.content,
                "token_usage": dict(usage) if usage else {},
                "generation_latency_ms": latency_ms,
            }
        except Exception as e:
            # Fallback local response
            user_msg = next((m for m in messages if m["role"] == "user"), {"content": ""})["content"]
            answer = (
                "DSA-Sensei (offline mode):\n"
                "- I will still guide you step-by-step using retrieved notes.\n\n"
                "Step 1: Understand the question.\n"
                f"- {user_msg}\n\n"
                "Step 2: Key ideas you may need:\n"
                "- Patterns like Two Pointers, Sliding Window, Hashing, Sorting.\n"
                "- Mind the constraints and edge cases.\n\n"
                "Step 3: From context (if any), relevant snippets are provided above.\n\n"
                "Step 4: Outline a solution\n"
                "- Write down approach, then complexity O(n) / O(n log n) where applicable.\n\n"
                "Stay consistent — small daily practice builds mastery."
            )
            return {"answer": answer, "token_usage": {}, "generation_latency_ms": 0}

    def answer_question(self, user_id: str, question: str, k: int = 3) -> Dict[str, Any]:
        contexts = self._search(question, k=k)
        messages = self._compose_prompt(question, contexts)
        gen = self._answer(messages)
        return {
            "user_id": user_id,
            "question": question,
            "contexts": contexts,
            "answer": gen["answer"],
            "metrics": {
                "token_usage": gen.get("token_usage", {}),
                "generation_latency_ms": gen.get("generation_latency_ms", 0),
            },
        }
