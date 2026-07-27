from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class _Env(BaseSettings):
    qdrant_mode: Literal["remote", "local"] = Field("remote", env="QDRANT_MODE")
    qdrant_url: str = Field(..., env="QDRANT_URL")
    qdrant_port: int = Field(6333, env="QDRANT_PORT")
    qdrant_authorization: str = Field("", env="QDRANT_AUTHORIZATION")
    model_server_url: str = Field("http://localhost:7997", env="MODEL_SERVER_URL")
    embedding_model: str = Field("nomic-ai/nomic-embed-text-v2-moe", env="EMBED_MODEL")
    reranking_model: str = Field("jinaai/jina-reranker-v2-base-multilingual", env="RERANK_MODEL")


@lru_cache
def _get_env() -> _Env:
    return _Env()


# Proxy object — reads env vars on first access, not at import
class _EnvProxy:
    def __getattr__(self, name: str):
        return getattr(_get_env(), name)


Environment = _EnvProxy()