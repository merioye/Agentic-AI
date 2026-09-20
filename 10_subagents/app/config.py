from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    google_api_key: str = Field(default="", min_length=10)
    supervisor_model: str = Field(default="gemini-3.5-flash-lite")
    subagent_model: str = Field(default="gemini-3.5-flash-lite")

    # API auth: a simple static key for this tutorial. In real production
    # you'd swap this for OAuth2 / JWT / a proper identity provider.
    api_key: str = Field(default="", min_length=10)

    # Safety limits
    max_agent_steps: int = Field(default=15)            # recursion_limit passed to every agent
    request_timeout_seconds: int = Field(default=30)

    langsmith_tracing: bool = Field(default=False)
    langsmith_api_key: str = Field(default="", min_length=10)
    langsmith_project: str = Field(default="Agentic AI")

    log_level: str = Field(default="INFO")


settings = Settings()