# Example API responses

Captured with `EMBEDDING_BACKEND=hash` and `RERANK_BACKEND=feature` (same as CI).

| File | Endpoint |
| --- | --- |
| `query_response.json` | `POST /v1/query` (GPS-tracker SK-12) |
| `eval_response.json` | `GET /v1/eval/recall-at-5` |
| `meta.json` | `GET /v1/meta` |

Regenerate:

```bash
# from repo root, with hash/feature env
python -c "..."  # or run TestClient with lifespan as in CI
```
