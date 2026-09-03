from functools import lru_cache

import numpy as np

from app.config import get_settings
from app.domain.ranking import Chunk, feature_rerank

CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self._ce = None
        if backend == "cross_encoder":
            try:
                from sentence_transformers import CrossEncoder

                self._ce = CrossEncoder(CROSS_ENCODER_NAME)
            except Exception:
                self.backend = "feature"
                self._ce = None

    def rerank(self, query: str, chunks: list[Chunk], query_vec: np.ndarray) -> list[Chunk]:
        if not chunks:
            return []
        if self._ce is None:
            return feature_rerank(query, chunks, query_vec)
        pairs = [(query, f"{c.title}. {c.text}") for c in chunks]
        scores = np.asarray(self._ce.predict(pairs), dtype=np.float32)
        order = np.argsort(-scores)
        return [chunks[int(i)] for i in order]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker(get_settings().rerank_backend)
