# DSA-Sensei

A friendly DSA tutor with Retrieval-Augmented Generation (RAG). It supports:

- Super simple chat UI (React + Vite)
- Light/Dark themes with an olive/grey/beige palette
- Single super user: Charan (the app ignores other user IDs)
- RAG over local notes in `data/` using Qdrant (optional) or filesystem fallback
- Sentry integration (optional)

---

## Features

- **Single user (Charan)**: The backend enforces `user_id = "Charan"`.
- **Local embeddings by default**: `sentence-transformers/all-MiniLM-L6-v2` to avoid API quotas.
- **Offline-friendly**: If Qdrant is empty or ingestion hasn’t run, the backend scans `data/` and returns context snippets.
- **Theme toggle**: Light and dark themes with the requested palette.
- **Postman-first**: All endpoints documented for Postman.

---

## Project structure

- `frontend/`: React + Vite app
- `backend/`: FastAPI app
  - `app/main.py`: FastAPI routes
  - `app/services/rag_service.py`: Retrieval + generation logic
  - `app/db/ingest_data.py`: Optional ingestion into Qdrant
- `infra/`: Docker Compose for Postgres, Qdrant, Backend
- `data/`: Your DSA notes (`.txt`, `.md`)

---

## Requirements

- Node 18+
- Python 3.11+
- Docker (optional but recommended for Qdrant/Postgres)

---

## Environment

Create `.env` in repo root (you can copy `.env.example`):

```
# Database (optional)
DATABASE_URL=postgresql://postgres:example@localhost:5433/dsasensei_db

# Qdrant
QDRANT_URL=http://localhost:6333

# Sentry (optional)
SENTRY_DSN=

# Embeddings (local is default)
EMBEDDING_PROVIDER=local
LOCAL_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# If you want OpenAI
OPENAI_API_KEY=
```

- Frontend always sends `user_id: "Charan"`; backend also enforces `"Charan"`.

---

## Running

### Option A: Docker (recommended for infra)

```
cd infra
docker compose down -v
docker compose up --build
```

- Qdrant UI: http://localhost:6333/dashboard
- Backend: http://localhost:8000

### Option B: Local backend + Docker Qdrant

```
cd infra
docker compose up -d qdrant dsasensei-postgres

# in another terminal at repo root
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt
uvicorn app.main:app --reload --app-dir backend
```

---

## Frontend

```
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

Theme toggle is in the header. Colors:
- Dark: olive green accents (#708238), dark greys, light text
- Light: beige background (#f5f0e6), light green accents (#8fbf7f)

---

## Postman tests

- Sentry test (intentional error):
  - GET/POST `http://localhost:8000/sentry-debug`
  - Expect 500 and an event in Sentry if DSN is set.

- Ask:
  - POST `http://localhost:8000/ask`
  - Body:
    ```json
    { "user_id": "anything", "question": "Tell me about binary search" }
    ```
  - Backend returns contexts + answer. If OpenAI quota is exhausted, a local fallback answer is returned.

---

## Data and Retrieval

Place `.txt` or `.md` files under `data/` (e.g., `data/binary_search.txt`).

The backend retrieves context in this order:
1) Vector search from Qdrant (if ingested)
2) Filesystem fallback: scans `data/` and returns snippets that match the query tokens (prioritizes filename matches)

This means you can get useful context even if ingestion has not run.

---

## Optional: Ingest into Qdrant (low-RAM)

Local embeddings by default (no OpenAI):

```
source venv/bin/activate
EMBEDDING_PROVIDER=local INGEST_BATCH=1 INGEST_MAX_CHUNKS=0 INGEST_SLEEP_MS=200 \
python backend/app/db/ingest_data.py
```

- If your machine is low on RAM and the OS kills the process (exit 137), try:
  - Close memory-heavy apps
  - Keep only a single small file in `data/` first
  - Reduce chunk size/overlap in `ingest_data.py`
  - Increase `INGEST_SLEEP_MS`

Note: Ingestion is optional because of the filesystem fallback.

---

## Troubleshooting

- **Blank contexts / empty retrieval**
  - Ensure your `data/` files exist and contain relevant text.
  - Backend fallback scans `data/` if Qdrant has no vectors.

- **Exit 137 (killed) during ingestion**
  - OS out of memory. Use `INGEST_BATCH=1`, `INGEST_SLEEP_MS=200`, close heavy apps.
  - Ingestion is optional; fallback uses filesystem.

- **OpenAI 429 insufficient_quota**
  - The app uses local embeddings by default.
  - Generation falls back to a deterministic local response if OpenAI is not available.

- **Port already in use**
  - Stop duplicate servers or change the port: `uvicorn ... --port 8001`.

---

## Deployment

### Frontend (GitHub Pages)

1. Set the backend API base to your deployed backend URL in the frontend (replace `http://localhost:8000/ask` inside `frontend/src/App.tsx`).
2. Build:
   ```bash
   cd frontend
   npm install
   npm run build
   ```
3. Deploy `frontend/dist/` to GitHub Pages (via GitHub Actions or Pages settings). Alternatively use Netlify/Vercel as a static host.

### Backend (Render / Railway / Fly.io)

- Easiest path: Render or Railway with Docker.

Render (example):
1. Push this repo to GitHub.
2. Create a new Render Web Service.
3. Use Docker, context at repo root, Dockerfile at `backend/Dockerfile`.
4. Env vars: `QDRANT_URL` (managed Qdrant) or deploy Qdrant as another service.
5. Expose port `8000`.

Railway (example):
1. New project → service from repo.
2. Use `backend/Dockerfile`, set envs.
3. Deploy; you’ll get a public URL like `https://your-app.up.railway.app`.

Qdrant options:
- Managed Qdrant (Qdrant Cloud) and set `QDRANT_URL`.
- Deploy a Qdrant service alongside backend.
- Or rely on filesystem fallback (no vector DB needed to get started).

Update the frontend to point to your deployed backend API.

---

## License

MIT