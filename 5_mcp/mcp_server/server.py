"""
DevOps MCP server.
Exposes three tools, one resource, and one prompt template over MCP.

Run modes:
    python server.py
        Local stdio development.

    MCP_TRANSPORT=http python server.py
        Streamable HTTP on port 8001.

Authentication:
    If DEVOPS_MCP_TOKEN is configured, Streamable HTTP requests must
    provide the token as a Bearer token.

    stdio connections are implicitly trusted because the client
    spawned the MCP server process directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl, BaseModel, Field
from dotenv import load_dotenv

import deployment_registry as registry


load_dotenv(Path(__file__).resolve().with_name(".env"))


_required_token = os.environ.get("DEVOPS_MCP_TOKEN")

_resource_server_url = os.environ.get(
    "MCP_RESOURCE_SERVER_URL",
    "http://localhost:8001/mcp",
)

_issuer_url = os.environ.get(
    "MCP_ISSUER_URL",
    "https://internal.local",
)


if _required_token:

    class StaticTokenVerifier(TokenVerifier):
        """Verify the development static bearer token."""

        async def verify_token(
            self,
            token: str,
        ) -> AccessToken | None:
            if token != _required_token:
                return None

            return AccessToken(
                token=token,
                client_id="internal-agents",
                scopes=[
                    "devops:read",
                    "devops:write",
                ],
                resource=_resource_server_url,
                subject="internal-agents",
            )

    mcp = FastMCP(
        "DevOps MCP Server",
        token_verifier=StaticTokenVerifier(),
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(_issuer_url),
            resource_server_url=AnyHttpUrl(_resource_server_url),
            required_scopes=[
                "devops:read",
                "devops:write",
            ],
            validate_token_resource=True,
        ),
        host="0.0.0.0",
        port=8001,
    )

else:
    mcp = FastMCP(
        "DevOps MCP Server",
        host="0.0.0.0",
        port=8001,
    )

# --------------------------------------------------------------------
# Tools - the model decides when to call these.
# --------------------------------------------------------------------


@mcp.tool()
def list_deployed_services() -> list[str]:
    """List every service currently tracked by the deployment system."""
    return registry.list_services()


@mcp.tool()
def get_deployment_status(service_name: str) -> dict:
    """Get the current deployment status of a service: version, health,
    and when it was last deployed. Call this before diagnosing an issue,
    not after - status often rules out half the possible causes."""
    return registry.get_status(service_name)


class RestartRequest(BaseModel):
    service_name: str = Field(description="Exact service name to restart")
    confirmed: bool = Field(
        description="Must be explicitly true. Never set this without the "
        "user having explicitly confirmed the restart in this conversation."
    )


@mcp.tool()
def restart_service(request: RestartRequest) -> dict:
    """Restart a service. DESTRUCTIVE - only call this after the user has
    explicitly confirmed they want the named service restarted. Never
    infer confirmation from context; ask directly if it wasn't given."""
    if not request.confirmed:
        raise ValueError("Restart was not explicitly confirmed by the user. Ask first.")
    return registry.restart(request.service_name)


# --------------------------------------------------------------------
# Resource - read-only reference data the application can pull in as
# context, addressed by URI rather than invoked as an action.
# --------------------------------------------------------------------


@mcp.resource("runbook://{service_name}")
def runbook(service_name: str) -> str:
    """Incident runbook for a given service."""
    return registry.get_runbook(service_name)


# --------------------------------------------------------------------
# Prompt - a reusable template a human user selects explicitly.
# --------------------------------------------------------------------


@mcp.prompt()
def incident_triage(service_name: str) -> str:
    """Triage an incident for a given service."""
    return (
        f"An incident was just reported for {service_name}. "
        f"Check its deployment status and runbook, then summarize the "
        f"most likely causes and a recommended next action."
    )


if __name__ == "__main__":
    transport = "streamable-http" if os.environ.get("MCP_TRANSPORT") == "http" else "stdio"
    if transport == "streamable-http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")