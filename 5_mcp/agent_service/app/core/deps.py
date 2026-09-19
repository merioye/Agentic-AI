from __future__ import annotations

from fastapi import Request

from app.core.config import Settings


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_agent(request: Request):
    return request.app.state.agent


def get_mcp_tools(request: Request) -> list:
    return request.app.state.tools


def get_tool_server_map(request: Request) -> dict:
    return request.app.state.tool_server_map