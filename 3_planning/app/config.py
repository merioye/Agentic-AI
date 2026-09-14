"""Centralized config."""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    primary_model: str = Field(default="gemini-3.5-flash-lite") # planner, replanner, todo-agent
    executer_model: str = Field(default="gemini-3.5-flash-lite") # per-step executer
    google_api_key: str = Field(default="", min_length=10)

    # --- Plan-and-execute limits ---
    max_replan_cycles: int = Field(default=4) # hard ceiling - same spirit as ModelCallLimitMiddleware
    max_plan_steps: int = Field(default=10)

    max_model_calls_per_turn: int = Field(default=15)
    allowed_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Observability ---
    langsmith_tracing_v2: bool = Field(default=True)
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com")
    langsmith_api_key: str = Field(default="", min_length=10)
    langsmith_project: str = Field(default="Agentic AI")


@lru_cache
def get_settings() -> Settings:
    return Settings()