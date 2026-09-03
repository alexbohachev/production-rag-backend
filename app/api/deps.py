from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from fastapi import Depends, Header, HTTPException, Request

from app.config import Settings, get_settings
from app.domain.ports import KnowledgeStore
from app.embeddings import Embedder, get_embedder
from app.infra.cache import Cache, MemoryCache, RedisCache
from app.infra.memory_store import MemoryStore
from app.infra.rate_limit import RateLimiter
from app.infra.resilience import CircuitBreaker
from app.services.query import QueryService

logger = logging.getLogger(__name__)


@dataclass
class Container:
    store: KnowledgeStore = field(default_factory=MemoryStore)
    cache: Cache = field(default_factory=MemoryCache)
    idempotency: dict[str, dict] = field(default_factory=dict)
    rate: RateLimiter = field(init=False)
    cache_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    def __post_init__(self) -> None:
        self.rate = RateLimiter(get_settings().rate_limit_per_minute)


container = Container()


def request_id_header(x_request_id: str | None = Header(default=None)) -> str:
    return x_request_id or str(uuid.uuid4())


def api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
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
        return container.cache

    async def _redis() -> RedisCache:
        cache = RedisCache(settings.redis_url)
        await cache.ping()
        return cache

    try:
        return await container.cache_breaker.call(_redis)
    except Exception:
        logger.warning("redis unavailable; falling back to memory cache")
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
