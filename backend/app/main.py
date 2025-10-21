import os
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from app.services.rag_service import RagService

load_dotenv()

app = FastAPI(title="DSA-Sensei", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional Sentry
try:
    import sentry_sdk  # type: ignore

    SENTRY_DSN = os.getenv("SENTRY_DSN")
    if SENTRY_DSN:
        sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=0.2)
except Exception:
    pass

rag_service = RagService()


class AskRequest(BaseModel):
    user_id: str
    question: str


@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    start = time.time()
    try:
        result = rag_service.answer_question(user_id=req.user_id, question=req.question)
        result["latency_ms"] = int((time.time() - start) * 1000)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
