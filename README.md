# Production RAG Backend

Hybrid retrieval HTTP service: **auth, hybrid search, rerank, structured citations, cache, metrics, Docker, CI**.

This is a **backend service**, not a Chat-with-PDF notebook. Corpus is synthetic ops knowledge (public demo). **Not AgriChain source.**

## Architecture

```
API  →  QueryService  →  store (memory | Postgres+pgvector)
                 ↘ cache (memory | Redis, fail-open)
                 ↘ embed (thread pool + timeout)
BM25 ∥ vector  →  RRF  →  lexical rerank  →  structured JSON
```

| Backend habit | Where |
| --- | --- |
| Layered code | `api/` `services/` `domain/` `infra/` |
| API design | `/v1/query`, `/v1/documents`, error envelope, `X-API-Key`, `X-Request-ID`, `Idempotency-Key` |
| Concurrency | `asyncio.gather` for BM25 + vector; embeddings in `to_thread` |
| Caching | query result cache; Redis optional, memory fallback |
| Resilience | embed timeout, Redis circuit breaker (fail-open) |
| Observability | Prometheus `/metrics`, request id |
| Tests + CI | pytest + ruff + GitHub Actions |
| Docker | Postgres (pgvector) + Redis + API |

## Run tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Run API (no Docker)

Default embeddings: **all-MiniLM-L6-v2**. First start downloads the model (~80 MB).

```bash
pip install -r requirements.txt -r requirements-ml.txt
copy .env.example .env
uvicorn app.main:app --reload
curl -s -H "X-API-Key: dev-key" localhost:8000/v1/meta
```

`/v1/meta` should show `"embedding": "minilm"`.

## Docker

```bash
docker compose up --build
```

`DATABASE_URL` and `REDIS_URL` are set in compose. Without them, the process uses in-memory store and cache (same API).

## Env

See `.env.example`. Local default is `EMBEDDING_BACKEND=minilm`. Tests and Docker stay on `hash` so CI/images stay small. If MiniLM fails to load, the process falls back to hash and logs the error.

## GitHub

Use **this folder as the repository root** so `.github/workflows/ci.yml` runs.
