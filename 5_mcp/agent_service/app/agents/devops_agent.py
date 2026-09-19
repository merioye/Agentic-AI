"""
The agent itself. Now how little agent-specific code there is here -
that's the point of MCP. The agent doesn't know or care that
list_deployed_services() lives in a subprocess we spawned, that
restart_service() requires a bearer token over HTTP, or that
read_file() came from an entirely separate vendor's server. It just
sees a flat list of LangChain-compatible tools.
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import Settings


SYSTEM_PROMPT = """You are an internal DevOps assistant. You have tools to
check deployment status, read incident runbooks, restart services, and
read files from the project directory.

Rules:
- Never call restart_service unless the user has explicitly confirmed
  they want that specific service restarted in this conversation.
- Check deployment status and the relevant runbook before diagnosing an
  issue - don't guess at causes.
- Be concise. This is an ops tool, not a chat companion."""


def build_devops_agent(settings: Settings, tools: list, checkpointer=None):
    model = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.google_api_key
    )

    return create_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
        system_prompt=SYSTEM_PROMPT
    )