# Eval protocol (synthetic)

`GET /v1/eval/recall-at-5` scores **this repo's** labeled set (`labels.json` + `corpus/docs.json`).

- `recall_at_1` / `recall_at_5` here are **not** AgriChain numbers.
- AgriChain: Recall@5 **72% vector → 88% hybrid** on **150** internal enterprise queries. That corpus is private.
- On this tiny public set, `@5` often saturates at 1.0. Use `@1` and the `cases` field to study BM25 vs vector.
