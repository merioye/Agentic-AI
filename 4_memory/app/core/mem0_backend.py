"""
Mem0-backend alternative to app/core/memory.py's semantic/episodic tools.

Why this exists as a separate module: Mem0 does its own fact extraction,
deduplication, and hybrid vector+graph retrieval, and benchmarks show it
returning results in ~0.2s p95 versus LangMem's reflection-heavy tools at
tens of seconds p95. For an interactive, latency-sensitive chat agent,
Mem0 (or a similar fast backend) is usually the better choice for the
hot path - reserve LangMem-style reflection for offline/background work.

Swap which backend the agent uses by changing MEMORY_BACKEND in .env;
app/agents/support_agent.py reads that flag and picks the tool set below
or the LangMem tool set from memory.py. The agent code itself doesn't care
which one is active.
"""
from __future__ import annotations

from langchain_core.tools import tool
from mem0 import Memory

_mem0_client: Memory | None = None


def get_mem0_client() -> Memory:
    global _mem0_client
    if _mem0_client is None:
        # Memory() runs fully local (SQLITE + local vector store) by default.
        # Fro hosted Mem0; from mem0 import MemoryClient; MemoryClient(api_key=...)
        _mem0_client = Memory()
    return _mem0_client


@tool
def mem0_remember(fact: str, user_id: str) -> str:
    """Store a fact or preference about the user for future conversations"""
    client = get_mem0_client()
    client.add([{"role": "user", "content": fact}], user_id=user_id)
    return "Saved."


@tool
def mem0_recall(query: str, user_id: str) -> str:
    """Search everything known about the user relevant to the query."""
    client = get_mem0_client()
    results = client.search(query, user_id=user_id)
    matches = results.get("results", [])
    if not matches:
        return "Nothing relevant found."
    return "\n".join(f"- {m['memory']}" for m in matches)


def mem0_memory_tools():
    return [mem0_remember, mem0_recall]