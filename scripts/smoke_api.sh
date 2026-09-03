#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE:-http://localhost:8000}"
KEY="${API_KEY:-dev-key}"
HDR=(-H "X-API-Key: $KEY")

curl -sf "$BASE/health" | tee /dev/stderr | grep -q ok
curl -sf "${HDR[@]}" "$BASE/ready" | tee /dev/stderr
curl -sf "${HDR[@]}" -H "Content-Type: application/json" \
  --data-binary @docs/examples/query_payload.json \
  "$BASE/v1/query" | tee /dev/stderr | grep -q citations
curl -sf "${HDR[@]}" "$BASE/v1/eval/recall-at-5" | tee /dev/stderr | grep -q recall_at_1
curl -sf "$BASE/metrics" | head -n 5
echo "smoke ok"
