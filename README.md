# Production RAG Backend

Hybrid retrieval HTTP service: **auth, hybrid search, feature rerank, grounded citations, cache, metrics, Docker, CI**.

Synthetic ops corpus. **Not AgriChain source.** Production AgriChain RAG uses Elasticsearch; this repo is the public backend shape (Postgres/pgvector or in-memory).

## Architecture

```
API → QueryService → store.bm25_ids ∥ store.vector_ids (SQL when Postgres)
                 → RRF → feature rerank (overlap + cosine)
                 → grounded answer + citations
                 ↘ Redis/memory cache (fail-open)
```

`GET /v1/eval/recall-at-5` runs the same Recall@5 protocol on **24 synthetic queries**. On this tiny corpus all modes hit 1.0 — that is saturation, not the AgriChain 150-query result (72% → 88%).

`Idempotency-Key` on `POST /v1/documents` replays the first result.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Run

```bash
pip install -r requirements.txt -r requirements-ml.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Local default embeddings: MiniLM. Docker/CI: `hash`.
