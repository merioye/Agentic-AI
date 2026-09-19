"""
Configuration for the agent host service. Two things live here that are
specific to MCP: the connection config for our own DevOps server, and
the connection config for whatever external server(s) we fan out to.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = Field(default="", min_length=10)
    chat_model: str = Field(default="gemini-3.5-flash-lite")

    app_name: str = Field(default="MCP DevOps Agent")
    environment: str = Field(default="development")

    # --- Our own DevOps MCP server ---
    # "stdio"   -> spawn mcp_server/server.py as a subprocess (simplest for local dev)
    # "http"    -> connect to an already-running server over streamable HTTP
    devops_mcp_transport: str = Field(default="stdio")
    devops_mcp_server_path: str = Field(default="../mcp_server/server.py")
    devops_mcp_url: str = Field(default="http://localhost:8001/mcp")
    devops_mcp_token: str = Field(default="")

    # --- External MCP server, filesystem as a stand-in for "someone else's tools" ---
    # Requires Node.js + npx available on PATH. Set enable=False to skip it
    # entirely (e.g. running in an environment without npx).
    enable_filesystem_server: bool = Field(default=True)
    filesystem_root: str = Field(default=".")


@lru_cache
def get_settings() -> Settings:
    return Settings()