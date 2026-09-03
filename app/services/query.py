from __future__ import annotations

import asyncio
import hashlib
import json
import logging

import numpy as np

from app.config import Settings
from app.domain.answer import grounded_answer
from app.domain.ports import KnowledgeStore
from app.domain.ranking import feature_rerank, rrf_fuse
from app.embeddings import Embedder
from app.infra.cache import Cache
from app.infra.resilience import CircuitOpenError, with_timeout
from app.schemas import QueryResponse, RetrievalTrace

logger = logging.getLogger(__name__)


class QueryService:
    def __init__(
        self,
        store: KnowledgeStore,
        embedder: Embedder,
        cache: Cache,
        settings: Settings,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.cache = cache
        self.settings = settings

    def _cache_key(self, query: str, top_k: int) -> str:
        raw = f"{self.embedder.backend}:{top_k}:{query.strip().lower()}"
        return "q:" + hashlib.sha256(raw.encode()).hexdigest()

    async def retrieve(
        self, query: str, top_k: int
    ) -> tuple[list, RetrievalTrace]:
        retrieve_k = max(self.settings.retrieve_k, top_k)
        query_vec = await self._embed(query)
        bm25_ids, vector_ids = await asyncio.gather(
            self.store.bm25_ids(query, retrieve_k),
            self.store.vector_ids(query_vec, retrieve_k),
        )
        fused = rrf_fuse(bm25_ids, vector_ids)[:retrieve_k]
        fused_chunks = await self.store.get_many(fused)
        reranked = feature_rerank(query, fused_chunks, query_vec)[:top_k]
        trace = RetrievalTrace(
            bm25_ids=bm25_ids[:10],
            vector_ids=vector_ids[:10],
            fused_ids=fused[:10],
            reranked_ids=[c.id for c in reranked],
        )
        return reranked, trace

    async def query(self, query: str, top_k: int, include_trace: bool) -> QueryResponse:
        key = self._cache_key(query, top_k)
        cached = await self._cache_get(key)
        if cached is not None:
            payload = json.loads(cached)
            if not include_trace:
                payload["retrieval"] = None
            return QueryResponse.model_validate(payload)

        if await self.store.count() == 0:
            return QueryResponse(answer="Index is empty.", citations=[], confidence=0.0)

        reranked, trace = await self.retrieve(query, top_k)
        answer, citations, confidence = grounded_answer(query, reranked)
        response = QueryResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            retrieval=trace,
        )
        if citations:
            await self._cache_set(key, response.model_dump_json())
        if not include_trace:
            response = response.model_copy(update={"retrieval": None})
        return response

    async def recall_at_5(self, labels: dict) -> dict[str, float]:
        modes = {"bm25": 0.0, "vector": 0.0, "hybrid": 0.0, "reranked": 0.0}
        n = 0
        for row in labels.values():
            query = row["query"]
            relevant = set(row["relevant_ids"])
            reranked, trace = await self.retrieve(query, 5)
            n += 1
            modes["bm25"] += _hit(trace.bm25_ids[:5], relevant)
            modes["vector"] += _hit(trace.vector_ids[:5], relevant)
            modes["hybrid"] += _hit(trace.fused_ids[:5], relevant)
            modes["reranked"] += _hit([c.id for c in reranked][:5], relevant)
        if not n:
            return modes
        return {k: round(v / n, 3) for k, v in modes.items()}

    async def _embed(self, query: str) -> np.ndarray:
        async def _run() -> np.ndarray:
            vectors = await asyncio.to_thread(self.embedder.embed, [query])
            return vectors[0]

        return await with_timeout(_run(), self.settings.embed_timeout_seconds)

    async def _cache_get(self, key: str) -> str | None:
        try:
            return await self.cache.get(key)
        except (CircuitOpenError, Exception):
            logger.warning("cache get failed", extra={"key": key})
            return None

    async def _cache_set(self, key: str, value: str) -> None:
        try:
            await self.cache.set(key, value, self.settings.cache_ttl_seconds)
        except Exception:
            logger.warning("cache set failed", extra={"key": key})


def _hit(ids: list[str], relevant: set[str]) -> float:
    return 1.0 if relevant & set(ids) else 0.0
