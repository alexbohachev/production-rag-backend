# Як це вчити (українською)

Це не «чат з PDF». Це маленький **retrieval backend**. На інтерв’ю розказуй пайплайн, не список бібліотек.

## Що прогнати руками

```bash
pytest -q
pytest tests/test_hybrid_lesson.py -v
```

`test_hybrid_lesson.py` — головний урок:

1. **BM25** дивиться на токени. Парафраз без спільних слів програє лексичному дистрактору.
2. **Vector** дивиться на напрям ембедінга. Той самий парафраз знаходить правильний документ.
RRF: `1/(k+rank)` по кожному списку. Документ у **обох** списках отримує суму балів і часто виходить вище за «переможця» лише одного списку.

Потім:

```text
GET /v1/eval/recall-at-5
```

Дивись **`recall_at_1`**, не лише `@5`. На маленькому корпусі `@5` легко дає 1.0 — це насичення, не «кращий за прод». У проді AgriChain інші дані: 150 запитів, Recall@5 72% → 88%. Це репо лише **той самий протокол**.

Кейси `q25` / `q26` в `eval/labels.json` підписані полем `lesson`.

## Карта коду

| Файл | Навіщо |
| --- | --- |
| `app/domain/ranking.py` | BM25, cosine, RRF, feature rerank |
| `app/rerank.py` | `feature` (CI) або `cross_encoder` MiniLM; fallback логується |
| `app/infra/pg_store.py` | BM25/`tsvector` + vector/`<=>` у SQL; GIN + HNSW індекси |
| `app/services/query.py` | оркестрація: embed → два ретрівери паралельно → fuse → rerank → answer |
| `app/infra/memory_store.py` | те саме API, що Postgres |
| `app/infra/pg_store.py` | BM25 = `tsvector`, vector = `<=>` |
| `app/infra/resilience.py` | timeout, circuit breaker (Redis fail-open) |
| `app/api/routes.py` | контракт HTTP |

## Питання, які мають вилітати з язика

- Чому kNN без BM25 губить `GPS-tracker SK-12`?
- Чому BM25 губить «hydrate rapidly draining fields» vs «sandy loam irrigation»?
- Навіщо `asyncio.gather` на двох ретріверах?
- Що робить `Idempotency-Key` при повторному POST?
- Чому відповідь **grounded** (цитати), а не вільний LLM?

LangGraph у цьому репо немає — він у `agentic-ai-backend`.
