from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.domain.ports import KnowledgeStore
from app.domain.ranking import Chunk


class PgStore(KnowledgeStore):
    def __init__(self, database_url: str) -> None:
        self.engine: AsyncEngine = create_async_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
        )

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS chunks (
                        id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        text TEXT NOT NULL,
                        embedding vector(384) NOT NULL
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS chunks_fts_idx
                    ON chunks USING GIN (to_tsvector('english', title || ' ' || text))
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
                    ON chunks USING hnsw (embedding vector_cosine_ops)
                    """
                )
            )

    async def upsert(self, chunks: list[Chunk]) -> int:
        async with self.engine.begin() as conn:
            for chunk in chunks:
                await conn.execute(
                    text(
                        """
                        INSERT INTO chunks (id, doc_id, title, text, embedding)
                        VALUES (:id, :doc_id, :title, :text, CAST(:embedding AS vector))
                        ON CONFLICT (id) DO UPDATE SET
                            doc_id = EXCLUDED.doc_id,
                            title = EXCLUDED.title,
                            text = EXCLUDED.text,
                            embedding = EXCLUDED.embedding
                        """
                    ),
                    {
                        "id": chunk.id,
                        "doc_id": chunk.doc_id,
                        "title": chunk.title,
                        "text": chunk.text,
                        "embedding": "[" + ",".join(f"{x:.7f}" for x in chunk.embedding) + "]",
                    },
                )
        return len(chunks)

    async def all_chunks(self) -> list[Chunk]:
        async with self.engine.connect() as conn:
            result = await conn.execute(
                text("SELECT id, doc_id, title, text, embedding::text FROM chunks")
            )
            rows = result.mappings().all()
        return [_row_to_chunk(row) for row in rows]

    async def get_many(self, ids: Sequence[str]) -> list[Chunk]:
        if not ids:
            return []
        async with self.engine.connect() as conn:
            result = await conn.execute(
                text(
                    """
                    SELECT id, doc_id, title, text, embedding::text
                    FROM chunks
                    WHERE id = ANY(CAST(:ids AS text[]))
                    """
                ),
                {"ids": list(ids)},
            )
            found = {_row_to_chunk(row) for row in result.mappings().all()}
        by_id = {c.id: c for c in found}
        return [by_id[i] for i in ids if i in by_id]

    async def count(self) -> int:
        async with self.engine.connect() as conn:
            result = await conn.execute(text("SELECT COUNT(*) FROM chunks"))
            return int(result.scalar_one())

    async def bm25_ids(self, query: str, k: int) -> list[str]:
        """Lexical retrieval via Postgres FTS (tsvector / ts_rank_cd), not Okapi BM25."""
        async with self.engine.connect() as conn:
            result = await conn.execute(
                text(
                    """
                    SELECT id
                    FROM chunks
                    WHERE to_tsvector('english', title || ' ' || text)
                          @@ plainto_tsquery('english', :q)
                    ORDER BY ts_rank_cd(
                        to_tsvector('english', title || ' ' || text),
                        plainto_tsquery('english', :q)
                    ) DESC
                    LIMIT :k
                    """
                ),
                {"q": query, "k": k},
            )
            return [row[0] for row in result.all()]

    async def vector_ids(self, query_vec, k: int) -> list[str]:
        literal = "[" + ",".join(f"{float(x):.7f}" for x in query_vec.tolist()) + "]"
        async with self.engine.connect() as conn:
            result = await conn.execute(
                text(
                    """
                    SELECT id
                    FROM chunks
                    ORDER BY embedding <=> CAST(:vec AS vector)
                    LIMIT :k
                    """
                ),
                {"vec": literal, "k": k},
            )
            return [row[0] for row in result.all()]


def _row_to_chunk(row) -> Chunk:
    raw = row["embedding"]
    if isinstance(raw, str):
        vec = json.loads(raw)
    else:
        vec = list(raw)
    arr = np.asarray(vec, dtype=np.float32)
    return Chunk(
        id=row["id"],
        doc_id=row["doc_id"],
        title=row["title"],
        text=row["text"],
        embedding=tuple(float(x) for x in arr),
    )
