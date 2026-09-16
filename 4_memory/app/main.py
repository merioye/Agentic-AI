"""
Application entrypoint.

Resources that are expensive to create and safe to share across requests
(the store, checkpointer, and agent graph) are built once inside the
lifespan context manager and attached to app.state - this is the current
recommended FastAPI pattern.

Run with:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agents.support_agent import build_support_agent
from app.core.config import get_settings
from app.core.memory import build_checkpointer, build_store, ensure_playbook_seeded
from app.routers import chat, memory_admin


# Set LangSmith env vars before any langchain imports
settings = get_settings()

os.environ["LANGSMITH_TRACING_V2"] = str(settings.langsmith_tracing_v2).lower()
os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("support_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting up | backend=%s memory_backend=%s", settings.persistence_backend, settings.memory_backend)

    store = build_store(settings)
    checkpointer = build_checkpointer(settings)
    ensure_playbook_seeded(store)
    agent = build_support_agent(settings, checkpointer, store)

    app.state.settings = settings
    app.state.store = store
    app.state.checkpointer = checkpointer
    app.state.agent = agent

    yield # ---- app is running, serving request ----

    logger.info("Shutting down.")
    # If using PostgresSave/PostgresStore context managers in production,
    # close them here (checkpointer.__exit__(...) / store.__exit()__).


app = FastAPI(
    title="Support Agent",
    description="Agent with working, semantic, episodic, and procedural memory.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(chat.router)
app.include_router(memory_admin.router)

# Dev-only CORS: the static UI is served from this same FastAPI process
# below, so this mainly matters if you ever point a different frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if get_settings().environment == "development" else [],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Server the browser UI at http://localhost:8000/ - html=True makes
# StaticFiles fall back to index.html for the root path.
app.mount('/', StaticFiles(directory="app/static", html=True), name="static")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})

@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok"}