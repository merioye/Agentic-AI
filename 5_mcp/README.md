# MCP DevOps Agent — Server + Host Project

Two independent pieces, demonstrating both sides of MCP:

```
mcp_project/
├── mcp_server/        # the SERVER — exposes DevOps tools over MCP
└── agent_service/      # the HOST/CLIENT — a FastAPI agent that connects
                        #   to mcp_server AND an external filesystem server
```

## Why two services

This is the point of MCP: `agent_service` doesn't import anything from
`mcp_server`. It connects to it exactly the same way it connects to a
completely unrelated, third-party filesystem server — over a standard
protocol, with a flat list of tools at the end. You could delete
`mcp_server` entirely and swap in someone else's hosted MCP server by
changing three lines in `.env`, and the agent code wouldn't need to change
at all.

## Running it

**1. Start the DevOps MCP server first for HTTP mode**

By default `agent_service` spawns `mcp_server/server.py` itself as a
stdio subprocess, so you don't need to run it separately for local dev.
To run it as a standalone HTTP service instead (closer to a real remote
deployment):

```bash
cd mcp_server
uv sync
uv run python server.py
# now listening on http://localhost:8001/mcp
```

On Windows PowerShell, use:

```powershell
cd mcp_server
uv sync
uv run python server.py
```

The server reads `mcp_server/.env`, including `MCP_TRANSPORT=http` and
`DEVOPS_MCP_TOKEN`. Start this process before starting the agent.

If you do this, set in `agent_service/.env`:

```
DEVOPS_MCP_TRANSPORT=http
DEVOPS_MCP_URL=http://localhost:8001/mcp
DEVOPS_MCP_TOKEN=dev-secret-token
```

**2. Start the agent host service**

```bash
cd agent_service
uv sync
uv run uvicorn app.main:app --reload
```

On Windows PowerShell:

```powershell
cd agent_service
uv sync
Copy-Item .env.example .env   # fill in GOOGLE_API_KEY
uv run uvicorn app.main:app --reload
```

For stdio mode, set `DEVOPS_MCP_TRANSPORT=stdio` in `agent_service/.env`.
The agent starts the DevOps MCP server itself, so do not start `mcp_server`
in another terminal. The host uses its own uv environment to launch the
child process.

Visit `http://localhost:8000/` for the browser UI. The right-hand panel
lists every tool currently connected, tagged by which MCP server it came
from — you should see tools from both `devops` (amber) and `filesystem`
(blue).

> The filesystem server requires Node.js (`npx` on PATH). If you don't
> have it, set `ENABLE_FILESYSTEM_SERVER=false` in `.env` — the agent
> will run fine with just the devops server.

## Try it

- _"What services are deployed right now?"_ → calls `list_deployed_services` (devops)
- _"Is notifications-worker healthy?"_ → calls `get_deployment_status` (devops)
- _"What does the runbook say for notifications-worker?"_ → reads the `runbook://` resource (devops)
- _"List the files in this project"_ → calls a filesystem tool (external server)
- _"Restart notifications-worker"_ → the agent should ask for explicit
  confirmation before calling the destructive `restart_service` tool —
  this is enforced both in the system prompt and in the tool's own input
  schema (`confirmed: bool`), a belt-and-suspenders pattern worth reusing
  for any tool with real-world side effects.

## Endpoints (agent_service)

- `GET /tools` — every connected tool, with its origin server
- `POST /chat` — non-streaming chat turn
- `POST /chat/stream` — SSE-streamed chat turn (`token`, `tool_start`, `tool_end`, `tool_error`, `error`, `done`)
- `GET /health` — liveness check
- `GET /` — browser UI

## Swapping in a real external server

Edit `app/core/mcp_client.py`'s `build_server_config()`. Adding GitHub's
official server, for example:

```python
servers["github"] = {
    "transport": "streamable_http",
    "url": "https://api.githubcopilot.com/mcp/",
    "headers": {"Authorization": f"Bearer {settings.github_pat}"},
}
```

Nothing else in the project needs to change — the agent, the streaming
router, and the UI's tool panel all work off the flattened tool list
regardless of how many servers or what transports back it.

## Known simplifications

- `mcp_server`'s auth is a static bearer token for illustration —
  production should use full OAuth 2.0 (see the tutorial's notes on the
  2026-07-28 MCP spec's auth hardening).
- `InMemorySaver` is used for working memory here — swap to a persistent
  checkpointer before deploying, same as in the memory-agent project.
- `deployment_registry.py` is fake, in-memory data standing in for a real
  orchestration platform API.
