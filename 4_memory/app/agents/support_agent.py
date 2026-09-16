"""
The support agent itself: a single create_agent() call wiring together
- working memory            (checkpointer, injected at .invoke() time via thread_id)
- semantic/episodic memory  (tools, backend chosen by settings.memory_backend)
- procedural memory         (a dynamic system prompt built from the stored playbook)
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from app.core.config import Settings
from app.core.mem0_backend import mem0_memory_tools
from app.core.memory import episodic_memory_tools, get_playbook, semantic_memory_tools


SUPPORT_SYSTEM_PROMPT = """You are a customer support agent for Acme SaaS
Be concise, professional, and empathetic. Use your memory tools proactively:
- record durable facts about the user (plan tier, contact preference, etc.)
- record an episode once an issue is resolved
- search episodic memory before answering recurring-looking issues
Never fabricate account details you have not retrieved from a tool."""

@dynamic_prompt
def support_prompt(request: ModelRequest) -> str:
    """Build the system prompt using the current procedural memory."""

    store = request.runtime.store

    if store is None:
        return SUPPORT_SYSTEM_PROMPT

    playbook = get_playbook(store)

    return (
        f"{SUPPORT_SYSTEM_PROMPT}\n\n"
        "Operating rules learned so far:\n"
        f"{playbook}"
    )



def build_support_agent(
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
):
    if settings.memory_backend == "mem0":
        memory_tools = mem0_memory_tools()
    else:
        memory_tools = semantic_memory_tools()

    tools = [*memory_tools, *episodic_memory_tools()]

    model = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.google_api_key
    )

    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
        store=store,
        middleware=[support_prompt],
    )
    return agent