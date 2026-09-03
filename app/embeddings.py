from __future__ import annotations

import hashlib
import logging
from functools import lru_cache

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

MINILM_NAME = "all-MiniLM-L6-v2"


def _stable_hash_vec(text: str, dims: int) -> np.ndarray:
    vec = np.zeros(dims, dtype=np.float32)
    toks = text.lower().split()
    for i, tok in enumerate(toks):
        for n in (1, 2):
            gram = " ".join(toks[i : i + n])
            if not gram:
                continue
            digest = hashlib.sha256(gram.encode()).digest()
            idx = int.from_bytes(digest[:4], "little") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class Embedder:
    def __init__(self, backend: str, dims: int) -> None:
        self.backend = backend
        self.dims = dims
        self._model = None
        if backend == "minilm":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(MINILM_NAME)
                dim_fn = getattr(self._model, "get_embedding_dimension", None) or getattr(
                    self._model, "get_sentence_embedding_dimension"
                )
                self.dims = int(dim_fn())
                logger.info("embeddings: %s (%s dims)", MINILM_NAME, self.dims)
            except Exception:
                logger.exception("MiniLM failed to load; falling back to hash embeddings")
                self.backend = "hash"
                self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            return np.asarray(
                self._model.encode(texts, normalize_embeddings=True),
                dtype=np.float32,
            )
        return np.stack([_stable_hash_vec(t, self.dims) for t in texts])


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    return Embedder(s.embedding_backend, s.embedding_dims)
