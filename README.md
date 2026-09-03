# Production RAG Backend

Hybrid retrieval HTTP service: **auth, hybrid search, feature or cross-encoder rerank, grounded citations, cache, metrics, Docker, CI**.

Synthetic ops corpus. **Not AgriChain source.** Production AgriChain RAG uses Elasticsearch; this repo is the public backend shape (Postgres/pgvector or in-memory).

## Architecture

```
API → QueryService → store.bm25_ids ∥ store.vector_ids (SQL when Postgres)
                 → RRF → rerank (feature overlap+cosine, or MiniLM cross-encoder)
                 → grounded answer + citations
                 ↘ Redis/memory cache (fail-open)
```

`GET /v1/eval/recall-at-5` runs Recall@1 and Recall@5 on **this synthetic labeled set**. Numbers here are not AgriChain. Production AgriChain: 72% vector → 88% hybrid on 150 internal queries (`eval/README.md`).

`RERANK_BACKEND=feature` in tests/CI. Local MiniLM: `RERANK_BACKEND=cross_encoder` plus `requirements-ml.txt`.

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

Дивись **STUDY.md** — як розібрати BM25 vs vector vs hybrid.
