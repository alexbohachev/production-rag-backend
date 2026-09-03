import os

os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["RERANK_BACKEND"] = "feature"

from app.config import get_settings
from app.rerank import get_reranker

get_settings.cache_clear()
get_reranker.cache_clear()
