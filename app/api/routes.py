from fastapi import APIRouter, Depends, Header
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from app.api.deps import api_key, embedder_dep, query_service, settings_dep, store_dep
from app.config import Settings
from app.domain.ports import KnowledgeStore
from app.embeddings import Embedder
from app.schemas import IngestRequest, QueryRequest, QueryResponse
from app.services.ingest import IngestService
from app.services.query import QueryService

router = APIRouter()

QUERY_COUNT = Counter("rag_query_total", "Query requests", ["status"])
QUERY_LATENCY = Histogram("rag_query_seconds", "Query latency")
INGEST_COUNT = Counter("rag_ingest_documents_total", "Ingested documents")


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(store: KnowledgeStore = Depends(store_dep)) -> dict[str, object]:
    return {"status": "ready", "chunks": await store.count()}


@router.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/v1/documents", status_code=201)
async def ingest_documents(
    body: IngestRequest,
    store: KnowledgeStore = Depends(store_dep),
    embedder: Embedder = Depends(embedder_dep),
    _: str = Depends(api_key),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, object]:
    n = await IngestService(store, embedder).ingest(body.documents)
    INGEST_COUNT.inc(n)
    return {"upserted": n, "idempotency_key": idempotency_key}


@router.post("/v1/query", response_model=QueryResponse)
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


@router.get("/v1/meta")
async def meta(
    settings: Settings = Depends(settings_dep),
    embedder: Embedder = Depends(embedder_dep),
    _: str = Depends(api_key),
) -> dict[str, str]:
    return {
        "service": settings.app_name,
        "embedding": embedder.backend,
    }
