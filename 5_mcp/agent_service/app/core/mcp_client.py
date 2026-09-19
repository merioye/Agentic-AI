"""
This module is the single place that knows how to reach every MCP
server this host connects to. Swapping a server's transport, adding a
new external server, or rotating an auth token only touches this file
- app/agents/devops_agent.py just consumes whatever tool list comes out
of build_mcp_client().

This mirrors the same "isolate the integration surface" pattern used
for the memory backends in Day 5: agents should depend on capabilities
(a list of tools), not on how those tools are wired up.
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.config import Settings

logger = logging.getLogger("mcp_agent.client")


def _resolve_server_path(configured_path: str) -> str:
    path = Path(configured_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return str(path.resolve())


def build_server_config(settings: Settings) -> dict:
    servers: dict = {}

    # --- Our own DevOps server ---
    if settings.devops_mcp_transport == "http":
        headers = {}
        if settings.devops_mcp_token:
            headers["Authorization"] = f"Bearer {settings.devops_mcp_token}"
        servers["devops"] = {
            "transport": "streamable_http",
            "url": settings.devops_mcp_url,
            "headers": headers
        }
    else:
        servers["devops"] = {
            "transport": "stdio",
            "command": sys.executable,
            "args": [_resolve_server_path(settings.devops_mcp_server_path)]
        }

    # --- External server: filesystem, standing in for "someone else's tools" ---
    # In a real deployment you'd add Github, Slack, etc. here with the shape:
    # servers["github"] = {
    #   "transport": "streamable_http",
    #   "url": "https://api.githubcopilot.com/mcp/",
    #   "headers": {"Authorization": f"Bearer {settings.github_pat}"}
    # }
    if settings.enable_filesystem_server:
        filesystem_command = "npx"
        filesystem_args = ["-y", "@modelcontextprotocol/server-filesystem", settings.filesystem_root]
        if sys.platform == "win32":
            node_command = shutil.which("node") or "node"
            npx_cli = Path(node_command).parent / "node_modules" / "npm" / "bin" / "npx-cli.js"
            filesystem_command = node_command
            filesystem_args = [str(npx_cli), *filesystem_args]
        servers["filesystem"] = {
            "transport": "stdio",
            "command": filesystem_command,
            "args": filesystem_args
        }

    return servers


async def build_mcp_client(settings: Settings) -> MultiServerMCPClient:
    server_config = build_server_config(settings)
    logger.info("Connecting to MCP servers: %s", list(server_config.keys()))
    client = MultiServerMCPClient(server_config)
    return client


async def load_tools(client: MultiServerMCPClient) -> list:
    tools = await client.get_tools()
    logger.info("Loaded %d tools across all MCP servers: %s", len(tools), [t.name for t in tools])
    return tools


async def load_tools_with_server_map(
    client: MultiServerMCPClient, server_names: list[str]
) -> tuple[list, dict[str, str]]:
    """Fetch tools per-server so we can attribute each tool to its origin
    server for UI display. MultiServerMCPClient.get_tools() accepts an
    optional server_name filter; calling it once per configured server
    is a small, on-time startup cost that buys reliable attribution
    instead of guessing at tool metadata shape, which has changes across
    langchain-mcp-adapters versions."""
    all_tools: list = []
    server_map: dict[str, str] = {}


    for server_name in server_names:
        server_tools = await client.get_tools(server_name=server_name)
        for tool in server_tools:
            server_map[tool.name] = server_name
        all_tools.extend(server_tools)

    logger.info("Loaded %d tools: %s", len(all_tools), server_map)
    return all_tools, server_map