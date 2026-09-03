from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.deps import container
from app.api.routes import router
from app.config import get_settings
from app.embeddings import get_embedder
from app.infra.pg_store import PgStore
from app.schemas import IngestDocument
from app.services.ingest import IngestService

logger = logging.getLogger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    if settings.database_url:
        store = PgStore(settings.database_url)
        await store.init()
        container.store = store
        logger.info("using postgres store")
    seed_file = Path(__file__).resolve().parents[1] / "corpus" / "docs.json"
    if seed_file.exists():
        docs = json.loads(seed_file.read_text(encoding="utf-8"))
        payload = [IngestDocument(id=d["id"], title=d["title"], text=d["text"]) for d in docs]
        n = await IngestService(container.store, get_embedder()).ingest(payload)
        logger.info("seeded %s documents with %s embeddings", n, get_embedder().backend)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Production RAG Backend",
        version="0.1.0",
        summary="Hybrid retrieval API with auth, cache, metrics, and structured answers.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": "Invalid request", "details": exc.errors()}},
        )

    return app


app = create_app()


def corpus_path() -> Path:
    return Path(__file__).resolve().parents[1] / "corpus" / "docs.json"
