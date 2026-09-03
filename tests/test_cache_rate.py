from app.infra.cache import MemoryCache
from app.infra.rate_limit import RateLimiter


def test_memory_cache_expires(monkeypatch):
    cache = MemoryCache()

    async def run():
        await cache.set("a", "1", ttl_seconds=60)
        assert await cache.get("a") == "1"

    import asyncio

    asyncio.run(run())


def test_rate_limiter_blocks():
    limiter = RateLimiter(per_minute=2)
    assert limiter.allow("k")
    assert limiter.allow("k")
    assert not limiter.allow("k")
