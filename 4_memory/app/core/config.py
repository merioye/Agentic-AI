"""
Centralized application configuration.
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM Provider ---
    google_api_key: str = Field(default="", min_length=10)
    chat_model: str = Field(default="gemini-3.5-flash-lite")
    embedding_model: str = Field(default="gemini-embedding-001")
    embedding_dims: int = Field(default=1536)

    # --- Persistence ---
    # "memory"  -> InMemoryStore / InMemorySaver (dev only, wiped on restart)
    # "postgres"-> production-grade persistent store + checkpointer
    persistence_backend: str = Field(default="memory")
    postgres_dsn: str = Field(default="")

    # --- Long-term memory ---
    # "langmem" -> LangGraph Store + LangMem tools (single persistence layer)
    # "mem0"    -> Mem0 (faster interactive recall, own extraction pipeline)
    memory_backend: str = Field(default="langmem")

    # --- App ---
    app_name: str = Field(default="Support Agent")
    environment: str = Field(default="development")

    # --- Observability ---
    langsmith_tracing_v2: bool = Field(default=True)
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com")
    langsmith_api_key: str = Field(default="", min_length=10)
    langsmith_project: str = Field(default="Agentic AI")


@lru_cache
def get_settings() -> Settings:
    return Settings()
