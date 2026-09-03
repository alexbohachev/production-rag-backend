"""The lesson this repo exists to teach: BM25 and vectors fail differently; hybrid unions them."""

import numpy as np

from app.domain.ranking import Chunk, bm25_rank, rrf_fuse, vector_rank


def _chunk(cid: str, text: str, vec: list[float]) -> Chunk:
    v = np.asarray(vec, dtype=np.float32)
    v = v / (np.linalg.norm(v) or 1.0)
    return Chunk(id=cid, doc_id=cid, title=cid, text=text, embedding=tuple(float(x) for x in v))


def test_bm25_misses_paraphrase_without_shared_tokens():
    relevant = _chunk("rel", "Sandy loam irrigation schedule uses drip lines.", [1.0, 0.0, 0.0])
    distractor = _chunk("lex", "How often should I hydrate rapidly draining fields extra extra.", [0.0, 1.0, 0.0])
    query = "How often should I hydrate rapidly draining fields?"
    ids = bm25_rank(query, [relevant, distractor])
    assert ids[0] == "lex"


def test_vector_finds_paraphrase_when_tokens_differ():
    relevant = _chunk("rel", "Sandy loam irrigation schedule uses drip lines.", [1.0, 0.0, 0.0])
    distractor = _chunk("lex", "How often should I hydrate rapidly draining fields extra extra.", [0.0, 1.0, 0.0])
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ids = vector_rank(query_vec, [relevant, distractor])
    assert ids[0] == "rel"


def test_rrf_keeps_exact_token_and_paraphrase():
    fused = rrf_fuse(["lex"], ["rel"])
    assert set(fused) == {"lex", "rel"}
