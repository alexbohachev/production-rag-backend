import json
from pathlib import Path

from fastapi import APIRouter, Depends, Header
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from app.api.deps import (
    api_key,
    cache_dep,
    container,
    embedder_dep,
    query_service,
    settings_dep,
    store_dep,
)
from app.config import Settings
from app.domain.ports import KnowledgeStore
from app.embeddings import Embedder
from app.infra.cache import Cache
from app.rerank import get_reranker
from app.schemas import IngestRequest, QueryRequest, QueryResponse
from app.services.ingest import IngestService
from app.services.query import QueryService

router = APIRouter()

QUERY_COUNT = Counter("rag_query_total", "Query requests", ["status"])
QUERY_LATENCY = Histogram("rag_query_seconds", "Query latency")
INGEST_COUNT = Counter("rag_ingest_documents_total", "Ingested documents")


@router.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", tags=["ops"])
async def ready(
    store: KnowledgeStore = Depends(store_dep),
    cache: Cache = Depends(cache_dep),
) -> dict[str, object]:
    redis_ok = True
    try:
        redis_ok = await cache.ping()
    except Exception:
        redis_ok = False
    return {
        "status": "ready",
        "chunks": await store.count(),
        "store": container.store_backend,
        "cache": container.cache_backend,
        "cache_ping": redis_ok,
        "idempotency": container.idempotency.backend,
    }


@router.get("/metrics", tags=["ops"])
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/v1/documents", status_code=201, tags=["ingest"])
async def ingest_documents(
    body: IngestRequest,
    store: KnowledgeStore = Depends(store_dep),
    embedder: Embedder = Depends(embedder_dep),
    _: str = Depends(api_key),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, object]:
    if idempotency_key:
        cached = await container.idempotency.get(idempotency_key)
        if cached is not None:
            return cached
    n = await IngestService(store, embedder).ingest(body.documents)
    INGEST_COUNT.inc(n)
    payload = {"upserted": n, "idempotency_key": idempotency_key, "replayed": False}
    if idempotency_key:
        replay = {**payload, "replayed": True}
        await container.idempotency.put(idempotency_key, replay)
    return payload


@router.post("/v1/query", response_model=QueryResponse, tags=["query"])
async def query_documents(
    body: QueryRequest,
    service: QueryService = Depends(query_service),
    _: str = Depends(api_key),
) -> QueryResponse:
    with QUERY_LATENCY.time():
        try:
            result = await service.query(body.query, body.top_k, body.include_trace)
            QUERY_COUNT.labels("ok").inc()
            return result
        except Exception:
            QUERY_COUNT.labels("error").inc()
            raise


@router.get("/v1/eval/recall-at-5", tags=["eval"])
async def recall_at_5(
    service: QueryService = Depends(query_service),
    _: str = Depends(api_key),
) -> dict[str, object]:
    labels_path = Path(__file__).resolve().parents[2] / "eval" / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    report = await service.evaluate(labels)
    report["corpus"] = "synthetic-ops"
    report["note"] = (
        "Synthetic labeled set only. Numbers here are not AgriChain production metrics. "
        "See eval/README.md for the private 150-query protocol note."
    )
    return report


@router.get("/v1/meta", tags=["ops"])
async def meta(
    settings: Settings = Depends(settings_dep),
    embedder: Embedder = Depends(embedder_dep),
    _: str = Depends(api_key),
) -> dict[str, str]:
    return {
        "service": settings.app_name,
        "embedding": embedder.backend,
        "rerank": get_reranker().backend,
        "rerank_configured": settings.rerank_backend,
        "store": container.store_backend,
        "cache": container.cache_backend,
        "idempotency": container.idempotency.backend,
    }
