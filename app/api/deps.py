from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from fastapi import Depends, Header, HTTPException, Request
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings
from app.domain.ports import KnowledgeStore
from app.embeddings import Embedder, get_embedder
from app.infra.cache import Cache, MemoryCache, RedisCache
from app.infra.idempotency import IdempotencyStore, MemoryIdempotency, RedisIdempotency, build_idempotency
from app.infra.memory_store import MemoryStore
from app.infra.rate_limit import RateLimiter
from app.infra.resilience import CircuitBreaker
from app.services.query import QueryService

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass
class Container:
    store: KnowledgeStore = field(default_factory=MemoryStore)
    cache: Cache = field(default_factory=MemoryCache)
    idempotency: IdempotencyStore = field(default_factory=MemoryIdempotency)
    rate: RateLimiter = field(init=False)
    cache_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    store_backend: str = "memory"
    cache_backend: str = "memory"

    def __post_init__(self) -> None:
        self.rate = RateLimiter(get_settings().rate_limit_per_minute)


container = Container()


def configure_runtime(settings: Settings) -> None:
    container.idempotency = build_idempotency(settings.redis_url)
    if settings.database_url:
        container.store_backend = "postgres"
    else:
        container.store_backend = "memory"
    container.cache_backend = "redis" if settings.redis_url else "memory"


def request_id_header(x_request_id: str | None = Header(default=None)) -> str:
    return x_request_id or str(uuid.uuid4())


async def api_key(
    x_api_key: str | None = Depends(API_KEY_HEADER),
    settings: Settings = Depends(get_settings),
) -> str:
    if not x_api_key or x_api_key not in settings.api_key_set:
        raise HTTPException(
            status_code=401,
            detail={"code": "unauthorized", "message": "Invalid API key"},
        )
    if not container.rate.allow(x_api_key):
        raise HTTPException(
            status_code=429,
            detail={"code": "rate_limited", "message": "Too many requests"},
        )
    return x_api_key


def settings_dep() -> Settings:
    return get_settings()


def store_dep() -> KnowledgeStore:
    return container.store


def embedder_dep() -> Embedder:
    return get_embedder()


async def cache_dep(settings: Settings = Depends(get_settings)) -> Cache:
    if not settings.redis_url:
        container.cache_backend = "memory"
        return container.cache

    async def _redis() -> RedisCache:
        cache = RedisCache(settings.redis_url)
        await cache.ping()
        return cache

    try:
        cache = await container.cache_breaker.call(_redis)
        container.cache_backend = "redis"
        if isinstance(container.idempotency, MemoryIdempotency):
            container.idempotency = RedisIdempotency(cache)
        return cache
    except Exception:
        logger.warning("redis unavailable; falling back to memory cache")
        container.cache_backend = "memory"
        return container.cache


def query_service(
    store: KnowledgeStore = Depends(store_dep),
    embedder: Embedder = Depends(embedder_dep),
    cache: Cache = Depends(cache_dep),
    settings: Settings = Depends(get_settings),
) -> QueryService:
    return QueryService(store=store, embedder=embedder, cache=cache, settings=settings)


def bind_request_id(request: Request, rid: str = Depends(request_id_header)) -> None:
    request.state.request_id = rid
