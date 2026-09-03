import pytest
from app.infra.resilience import CircuitBreaker, CircuitOpenError, with_timeout


@pytest.mark.asyncio
async def test_timeout():
    async def slow():
        import asyncio

        await asyncio.sleep(1)
        return 1

    with pytest.raises(TimeoutError):
        await with_timeout(slow(), 0.01)


@pytest.mark.asyncio
async def test_circuit_opens_after_failures():
    br = CircuitBreaker(failure_threshold=2, reset_seconds=30)

    async def boom():
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        await br.call(boom)
    with pytest.raises(RuntimeError):
        await br.call(boom)
    with pytest.raises(CircuitOpenError):
        await br.call(boom)
