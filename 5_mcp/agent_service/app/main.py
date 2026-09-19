"""
Application entrypoint.

The important lifecycle detail: MCP connections (especially stdio
subprocesses) are set up once at startup and reused for every request -
NOT reconnected per-request. Reconnecting per-request would respawn a
subprocess (or re-handshake an HTTP session) on every single chat
message, which is both slow and wasteful. langchain-mcp-adapters'
MultiServerMCPClient is designed to be a long-lived object exactly for
this reason, matching how we treated the store/checkpointer as
long-lived singletons in the memory-agent project.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.devops_agent import build_devops_agent
from app.core.config import get_settings
from app.core.mcp_client import build_mcp_client, build_server_config, load_tools_with_server_map
from app.routers import chat


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting up | environment=%s", settings.environment)

    server_config = build_server_config(settings)
    mcp_client = await build_mcp_client(settings)
    tools, tool_server_map = await load_tools_with_server_map(mcp_client, list(server_config.keys()))

    checkpointer = InMemorySaver()
    agent = build_devops_agent(settings, tools, checkpointer=checkpointer)

    app.state.settings = settings
    app.state.mcp_client = mcp_client
    app.state.tools = tools
    app.state.tool_server_map = tool_server_map
    app.state.agent = agent

    yield # --- serving requests ---

    logger.info("Shutting down, closing MCP connections.")
    # stdio subprocess are torn down their connection context
    # closes; MultiServerMCPClient manages this internally. If you're on a
    # version that exposes and explicit close/aclose, call it here


app = FastAPI(
    title="MCP DevOps Agent",
    description="An MCP host connecting to custom internal server and an external server.",
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(chat.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(status_code=502, content={"detail": "Internal server error."})


@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")