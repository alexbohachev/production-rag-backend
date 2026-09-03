from app.domain.ranking import Chunk


class KnowledgeStore:
    async def upsert(self, chunks: list[Chunk]) -> int:
        raise NotImplementedError

    async def all_chunks(self) -> list[Chunk]:
        raise NotImplementedError

    async def get_many(self, ids: list[str]) -> list[Chunk]:
        raise NotImplementedError

    async def count(self) -> int:
        raise NotImplementedError
