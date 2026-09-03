import numpy as np

from app.domain.ranking import Chunk
from app.rerank import Reranker


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, doc_id=cid, title=cid, text=text, embedding=(0.0, 1.0))


def test_feature_backend_does_not_load_cross_encoder():
    r = Reranker("feature")
    assert r.backend == "feature"
    assert r._ce is None
    out = r.rerank("drip irrigation", [_chunk("a", "drip irrigation sandy loam"), _chunk("b", "unrelated")], np.ones(2))
    assert out[0].id == "a"


def test_cross_encoder_orders_by_pair_score():
    r = Reranker("feature")
    r.backend = "cross_encoder"

    class FakeCE:
        def predict(self, pairs):
            return [0.1 if "noise" in p[1] else 0.9 for p in pairs]

    r._ce = FakeCE()
    out = r.rerank(
        "hydrate fields",
        [_chunk("noise", "noise tokens GPS-tracker"), _chunk("gold", "drip irrigation sandy loam")],
        np.ones(2),
    )
    assert [c.id for c in out] == ["gold", "noise"]
