"""
Centralized config. EFFORT_LEVELS and the routing thresholds below are the
concrete implementation — dynamic effort routing is only as
good as these being tuned against real traffic, not left at guessed
defaults forever.
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Valid Gemini thinking effort levels, cheapest to most expensive
# Kept as an explicit list so app/reasoning.py can validate against
# it rather than accepting an arbitrary string.
EFFORT_LEVELS = ["minimal", "low", "medium", "high"]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    primary_model: str = Field(default="gemini-3.5-pro")
    fast_model: str = Field(default="gemini-3.5-flash-lite")
    google_api_key: str = Field(default="", min_length=10)

    # --- Effort routing ---
    # Heuristic first: these keywords bump the heuristic classification up
    # a tier before ever spending a classification call. Tune against your
    # own traffic - this list is a reasonable starting point, not gospel.
    high_effort_keywords: list[str] = Field(
        default_factory=lambda: [
            "prove", "calculate", "compare", "optimi", "constraint",
            "step by step", "derive", "multi-step"
        ]
    )

    # Above this character length, the heuristic assumes enough complexity
    # to warrant at least classifier call rather than defaulting minimal.
    heuristic_length_threshold: int = Field(default=280)
    use_llm_classifier_fallback: bool = Field(default=True)
    default_effort: str = Field(default="minimal")

    max_model_calls_per_turn: int  = Field(default=15)
    allowed_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Observability ---
    langsmith_tracing_v2: bool = Field(default=True)
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com")
    langsmith_api_key: str = Field(default="", min_length=10)
    langsmith_project: str = Field(default="Agentic AI")

@lru_cache
def get_settings() -> Settings:
    return Settings()
