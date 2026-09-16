"""
Dependency-injection accessors.

Resources built once at startup (agent, store, checkpointer, settings)
live on app.state, constructed inside the lifespan context manager in
main.py. These Depends() functions just hand them to route handlers -
this is the standard FastAPI pattern for sharing expensive, stateful
objects (DB connections, model clients) across requests without global
mutable state scattered through the codebase.
"""
from __future__ import annotations

from fastapi import Request
from langgraph.store.base import BaseStore

from app.core.config import Settings

def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> BaseStore:
    return request.app.state.store


def get_agent(request: Request):
    return request.app.state.agent