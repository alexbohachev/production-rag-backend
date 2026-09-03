from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.textutil import tokenize


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str
    embedding: tuple[float, ...]


def rrf_fuse(bm25_ids: list[str], vector_ids: list[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(bm25_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for rank, chunk_id in enumerate(vector_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


def bm25_rank(query: str, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> list[str]:
    q_terms = tokenize(query)
    if not q_terms or not chunks:
        return []
    docs = [tokenize(c.text) for c in chunks]
    avgdl = sum(len(d) for d in docs) / len(docs)
    df: dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    n = len(docs)
    scored: list[tuple[float, str]] = []
    for chunk, doc in zip(chunks, docs, strict=True):
        score = 0.0
        tf: dict[str, int] = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1
        dl = len(doc) or 1
        for term in q_terms:
            if term not in tf:
                continue
            n_q = df.get(term, 0)
            idf = np.log((n - n_q + 0.5) / (n_q + 0.5) + 1.0)
            denom = tf[term] + k1 * (1 - b + b * dl / avgdl)
            score += idf * (tf[term] * (k1 + 1) / denom)
        scored.append((float(score), chunk.id))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [i for s, i in scored if s > 0]


def vector_rank(query_vec: np.ndarray, chunks: list[Chunk]) -> list[str]:
    if not chunks:
        return []
    mat = np.asarray([c.embedding for c in chunks], dtype=np.float32)
    q = query_vec.astype(np.float32)
    qn = np.linalg.norm(q) or 1.0
    nn = np.linalg.norm(mat, axis=1)
    nn[nn == 0] = 1.0
    sims = mat @ (q / qn) / nn
    order = np.argsort(-sims)
    return [chunks[int(i)].id for i in order]


def lexical_rerank(query: str, chunks: list[Chunk]) -> list[Chunk]:
    q = set(tokenize(query))
    if not q:
        return chunks

    def score(chunk: Chunk) -> float:
        tokens = set(tokenize(chunk.title + " " + chunk.text))
        return len(q & tokens) / len(q)

    return sorted(chunks, key=score, reverse=True)


def feature_rerank(query: str, chunks: list[Chunk], query_vec: np.ndarray) -> list[Chunk]:
    """Mix token overlap with cosine — not a cross-encoder."""
    q = set(tokenize(query))
    qn = np.linalg.norm(query_vec) or 1.0
    qv = query_vec.astype(np.float32) / qn

    def score(chunk: Chunk) -> float:
        tokens = set(tokenize(chunk.title + " " + chunk.text))
        lex = (len(q & tokens) / len(q)) if q else 0.0
        emb = np.asarray(chunk.embedding, dtype=np.float32)
        en = np.linalg.norm(emb) or 1.0
        cos = float(emb @ qv / en)
        return 0.55 * lex + 0.45 * max(cos, 0.0)

    return sorted(chunks, key=score, reverse=True)
