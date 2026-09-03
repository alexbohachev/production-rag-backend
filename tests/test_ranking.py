from app.domain.ranking import Chunk, bm25_rank, lexical_rerank, rrf_fuse, vector_rank
from app.embeddings import Embedder


def _chunks() -> list[Chunk]:
    embedder = Embedder("hash", 384)
    docs = [
        ("doc-01", "Irrigation on sandy loam", "Sandy loam drains quickly. Water light soil more often."),
        ("doc-05", "GPS tracker SK-12 install", "The GPS-tracker SK-12 must sit on a metal roof."),
        ("doc-04", "Cercospora leaf spot on beet", "Cercospora threshold is 5 percent of mid-canopy leaves."),
    ]
    vecs = embedder.embed([f"{t}\n{b}" for _, t, b in docs])
    return [
        Chunk(id=i, doc_id=i, title=t, text=b, embedding=tuple(float(x) for x in vecs[n]))
        for n, (i, t, b) in enumerate(docs)
    ]


def test_bm25_prefers_exact_token():
    ids = bm25_rank("GPS-tracker SK-12 firmware", _chunks())
    assert ids[0] == "doc-05"


def test_vector_rank_nearest_self():
    chunks = _chunks()
    import numpy as np

    q = np.asarray(chunks[0].embedding, dtype=np.float32)
    ids = vector_rank(q, chunks)
    assert ids[0] == chunks[0].id


def test_rrf_keeps_both_lists():
    fused = rrf_fuse(["a", "b"], ["b", "c"])
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c"}


def test_lexical_rerank_puts_overlap_first():
    chunks = _chunks()
    ranked = lexical_rerank("Cercospora beet threshold", chunks)
    assert ranked[0].id == "doc-04"
