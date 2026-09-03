from app.main import create_app
from fastapi.testclient import TestClient


def test_health_is_public():
    with TestClient(create_app()) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"


def test_query_requires_api_key():
    with TestClient(create_app()) as client:
        res = client.post("/v1/query", json={"query": "When to irrigate sandy loam?"})
        assert res.status_code == 401


def test_query_returns_structured_citations():
    with TestClient(create_app()) as client:
        res = client.post(
            "/v1/query",
            json={"query": "Where to mount GPS-tracker SK-12?", "include_trace": True},
            headers={"X-API-Key": "dev-key"},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["citations"]
        assert body["confidence"] > 0
        assert "doc-05" in body["retrieval"]["reranked_ids"]
        assert res.headers.get("x-request-id")


def test_ingest_is_idempotent_shape():
    with TestClient(create_app()) as client:
        res = client.post(
            "/v1/documents",
            json={"documents": [{"id": "doc-x", "title": "T", "text": "hello world token token"}]},
            headers={"X-API-Key": "dev-key", "Idempotency-Key": "k1"},
        )
        assert res.status_code == 201
        assert res.json()["upserted"] == 1
