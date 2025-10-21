import os
import time
import uuid
from typing import Any, Dict, List

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from openai import OpenAI

load_dotenv()

COLLECTION_NAME = "dsa_docs"
EMBED_DIM = 1536  # text-embedding-3-small


class RagService:
    def __init__(self) -> None:
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=self.qdrant_url)
        self._ensure_collection()
        self.openai = OpenAI()

    def _ensure_collection(self) -> None:
        try:
            collections = self.client.get_collections()
            names = [c.name for c in collections.collections]
            if COLLECTION_NAME not in names:
                self.client.recreate_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=qmodels.VectorParams(
                        size=EMBED_DIM, distance=qmodels.Distance.COSINE
                    ),
                )
        except Exception:
            # Try to create regardless
            self.client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
            )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # OpenAI embeddings
        response = self.openai.embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in response.data]

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
