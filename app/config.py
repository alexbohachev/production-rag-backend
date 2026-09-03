from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "production-rag-backend"
    api_keys: str = "dev-key"
    database_url: str = ""
    redis_url: str = ""
    embedding_backend: str = "minilm"
    rerank_backend: str = "cross_encoder"  # feature | cross_encoder
    embedding_dims: int = 384
    retrieve_k: int = 20
    rerank_k: int = 5
    cache_ttl_seconds: int = 60
    embed_timeout_seconds: float = 30.0
    rate_limit_per_minute: int = 120
    log_json: bool = True

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
