from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str | None = Field(None)
    model_name: str | None = Field(None)
    small_model_name: str | None = Field(None)
    embedding_model_name: str | None = Field(None)
    # Separate base URL for embedding model (falls back to openai_base_url)
    embedding_base_url: str | None = Field(None)
    # Set to "true" to inject /no_think into system messages (for Qwen3.5, DeepSeek-R1, etc.)
    disable_llm_thinking: str | None = Field(None)
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


@lru_cache
def get_settings():
    return Settings()  # type: ignore[call-arg]


ZepEnvDep = Annotated[Settings, Depends(get_settings)]
