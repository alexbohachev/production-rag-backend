from __future__ import annotations

import asyncio
import hashlib
import json
import logging

import numpy as np

from app.config import Settings
from app.domain.answer import extractive_answer
from app.domain.ports import KnowledgeStore
from app.domain.ranking import bm25_rank, lexical_rerank, rrf_fuse, vector_rank
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

    async def query(self, query: str, top_k: int, include_trace: bool) -> QueryResponse:
        key = self._cache_key(query, top_k)
        cached = await self._cache_get(key)
        if cached is not None:
            payload = json.loads(cached)
            if not include_trace:
                payload["retrieval"] = None
            return QueryResponse.model_validate(payload)

        chunks = await self.store.all_chunks()
        if not chunks:
            return QueryResponse(answer="Index is empty.", citations=[], confidence=0.0)

        query_vec = await self._embed(query)
        retrieve_k = max(self.settings.retrieve_k, top_k)

        bm25_ids, vector_ids = await asyncio.gather(
            asyncio.to_thread(bm25_rank, query, chunks),
            asyncio.to_thread(vector_rank, query_vec, chunks),
        )
        bm25_ids = bm25_ids[:retrieve_k]
        vector_ids = vector_ids[:retrieve_k]
        fused = rrf_fuse(bm25_ids, vector_ids)[:retrieve_k]
        by_id = {c.id: c for c in chunks}
        fused_chunks = [by_id[i] for i in fused if i in by_id]
        reranked = lexical_rerank(query, fused_chunks)[:top_k]

        answer, citations, confidence = extractive_answer(query, reranked)
        trace = RetrievalTrace(
            bm25_ids=bm25_ids[:10],
            vector_ids=vector_ids[:10],
            fused_ids=fused[:10],
            reranked_ids=[c.id for c in reranked],
        )
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
