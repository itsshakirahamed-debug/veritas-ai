# Veritas Legal

Grounded legal assistant: FastAPI backend + minimal React/Vite frontend.
Every fact, claim, and citation is traceable to an indexed chunk with a verbatim quote — or it is reported as `[INFORMATION NOT IN RECORD]`.

## Run

```bash
# Backend (http://127.0.0.1:8000)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --port 8000

# Frontend (http://127.0.0.1:5173)
cd frontend && npm install && npm run dev
```

The UI ships with light/dark mode (toggle in the header; defaults to your system
preference and is remembered). Force a theme with `?theme=light` or `?theme=dark`.

Optional: `docker compose up -d` for Qdrant, then set `QDRANT_IN_MEMORY=false` to persist the index.
Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` to enable live LLM review/draft
(model configurable via `ANTHROPIC_MODEL`). Without a key, review/draft use a grounded
deterministic engine over the indexed chunks.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Status, doc/chunk counts, LLM availability |
| POST | `/documents/upload` | Ingest + index a PDF (multipart) |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{doc_id}` | Remove a document from the index |
| GET | `/documents/{doc_id}/chunks` | Chunks with page/offset metadata |
| GET | `/search?q=` | Hybrid search (vector + BM25, RRF fused) |
| POST | `/verify` | Check a quote is verbatim in its cited chunk |
| GET | `/source/{chunk_id}` | Full chunk text + offsets (404 if unknown) |
| POST | `/review` | Extracted facts + gap report (missing fields, contradictions) |
| POST | `/draft` | Grounded document draft + gap report |

## Tests

```bash
python -m pytest tests/ -q   # schemas, ingestion/indexing, retrieval/verify, API
cd frontend && npm run lint && npm run build
```
