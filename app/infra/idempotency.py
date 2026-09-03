from __future__ import annotations

import json
import logging
from typing import Protocol

from app.infra.cache import RedisCache

logger = logging.getLogger(__name__)


class IdempotencyStore(Protocol):
    async def get(self, key: str) -> dict | None: ...
    async def put(self, key: str, value: dict, ttl_seconds: int = 86400) -> None: ...
    @property
    def backend(self) -> str: ...


class MemoryIdempotency:
    def __init__(self) -> None:
        self._data: dict[str, dict] = {}
        self.backend = "memory"

    async def get(self, key: str) -> dict | None:
        return self._data.get(key)

    async def put(self, key: str, value: dict, ttl_seconds: int = 86400) -> None:
        self._data[key] = value


class RedisIdempotency:
    def __init__(self, cache: RedisCache) -> None:
        self._cache = cache
        self.backend = "redis"

    async def get(self, key: str) -> dict | None:
        raw = await self._cache.get(f"idem:{key}")
        if not raw:
            return None
        return json.loads(raw)

    async def put(self, key: str, value: dict, ttl_seconds: int = 86400) -> None:
        await self._cache.set(f"idem:{key}", json.dumps(value), ttl_seconds)


def build_idempotency(redis_url: str) -> IdempotencyStore:
    if not redis_url:
        return MemoryIdempotency()
    try:
        return RedisIdempotency(RedisCache(redis_url))
    except Exception:
        logger.exception("idempotency redis init failed; using memory")
        return MemoryIdempotency()
