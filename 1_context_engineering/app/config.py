"""
Centralized application configuration.

Loaded once at import time via pydantic-settings.  All environment-specific
values (API keys, model names, DB paths) live here - nothing else in the
app should call os.environ directly. This is the patterns you want at
production scale: one source of truth for config, validated at startup
instead of failing deep inside a request handler.
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config  = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider ---
    primary_model: str = Field(default="gemini-3.5-flash-lite")
    summarizer_model: str = Field(default="gemini-3.5-flash-lite")
    google_api_key: str = Field(default="", min_length=10)

    # --- Persistence ---
    # SQLite is fine for single-process demo / small production deployment.
    # Swap SQLITE_PATH usage for a Postgres connection string
    # (see docker-compose.yml) once you need multi-process / multi-node scale.
    sqlite_checkpoint_path: str = Field(default="./db/checkpoints.sqlite")
    app_db_path: str = Field(default="./db/app.sqlite")

    # --- API auth (demo only - see app/auth.py for production notes) ---
    demo_api_keys: dict[str, str] = Field(
        default_factory=lambda: {
            "guest-key": "guest",
            "customer-key": "customer",
            "admin-key": "admin"
        }
    )

    # --- Operational limits ---
    max_model_calls_per_turn: int = Field(default=25)
    summarize_after_tokens: int = Field(default=2000)
    request_timeout_seconds: float = Field(default=30.0)

    # --- CORS ---
    allowed_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Observability ---
    langsmith_tracing_v2: bool = Field(default=True)
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com")
    langsmith_api_key: str = Field(default="", min_length=10)
    langsmith_project: str = Field(default="Agentic AI")


@lru_cache
def get_settings() -> Settings:
    """Settings are cached - read once, reused for the life of the process."""
    return Settings()
