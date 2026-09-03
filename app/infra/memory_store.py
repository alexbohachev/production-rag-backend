from __future__ import annotations

import asyncio

from app.domain.ports import KnowledgeStore
from app.domain.ranking import Chunk


class MemoryStore(KnowledgeStore):
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, chunks: list[Chunk]) -> int:
        async with self._lock:
            for chunk in chunks:
                self._chunks[chunk.id] = chunk
            return len(chunks)

    async def all_chunks(self) -> list[Chunk]:
        async with self._lock:
            return list(self._chunks.values())

    async def get_many(self, ids: list[str]) -> list[Chunk]:
        async with self._lock:
            found = [self._chunks[i] for i in ids if i in self._chunks]
        by_id = {c.id: c for c in found}
        return [by_id[i] for i in ids if i in by_id]

    async def count(self) -> int:
        async with self._lock:
            return len(self._chunks)
