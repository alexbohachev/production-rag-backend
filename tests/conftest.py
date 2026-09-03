import os

os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["RERANK_BACKEND"] = "feature"

from app.api.deps import container
from app.config import get_settings
from app.infra.idempotency import MemoryIdempotency
from app.infra.memory_store import MemoryStore
from app.rerank import get_reranker

get_settings.cache_clear()
get_reranker.cache_clear()
container.store = MemoryStore()
container.idempotency = MemoryIdempotency()
container.store_backend = "memory"
container.cache_backend = "memory"
