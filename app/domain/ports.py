import numpy as np

from app.domain.ranking import Chunk, bm25_rank, vector_rank


class KnowledgeStore:
    async def upsert(self, chunks: list[Chunk]) -> int:
        raise NotImplementedError

    async def all_chunks(self) -> list[Chunk]:
        raise NotImplementedError

    async def get_many(self, ids: list[str]) -> list[Chunk]:
        raise NotImplementedError

    async def count(self) -> int:
        raise NotImplementedError

    async def bm25_ids(self, query: str, k: int) -> list[str]:
        return bm25_rank(query, await self.all_chunks())[:k]

    async def vector_ids(self, query_vec: np.ndarray, k: int) -> list[str]:
        return vector_rank(query_vec, await self.all_chunks())[:k]
