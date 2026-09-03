from __future__ import annotations

import asyncio

from app.domain.ports import KnowledgeStore
from app.domain.ranking import Chunk
from app.embeddings import Embedder
from app.schemas import IngestDocument


class IngestService:
    def __init__(self, store: KnowledgeStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    async def ingest(self, documents: list[IngestDocument]) -> int:
        texts = [f"{d.title}\n{d.text}" for d in documents]
        vectors = await asyncio.to_thread(self.embedder.embed, texts)
        chunks = [
            Chunk(
                id=doc.id,
                doc_id=doc.id,
                title=doc.title,
                text=doc.text,
                embedding=tuple(float(x) for x in vectors[i]),
            )
            for i, doc in enumerate(documents)
        ]
        return await self.store.upsert(chunks)
