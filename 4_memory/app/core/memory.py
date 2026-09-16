"""
Memory infrastructure for the agent.

This module is the single place that wires up all four CoALA memory types
onto LangGraph primitives:

  working memory        ->  checkpointer (thread_id scoped)
  semantic memory       ->  store, namespace ("semantic", user_id) via LangMem tools
  episodic memory       ->  store, namespace ("episodic", user_id) via custom tools
  procedural memory     -> store, namespace ("procedural", "global") via a playbook doc

Swapping InMemoryStore -> PostgresStore, or LangMem -> Mem0, only touches
this file. Nothing in the agent or API layer needs to change.
"""
from __future__ import annotations

import uuid
from typing import Annotated, cast

from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore, IndexConfig
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from pydantic import BaseModel, Field, SecretStr

from app.core.config import Settings


# --------------------------------------------------------------------------
# Namespaces - the naming convention that keeps memory types from colliding.
# A namespace is a tuple; LangGraph's store partitions data by it.
# --------------------------------------------------------------------------

def semantic_namespace(user_id: str) -> tuple[str, ...]:
    return ("semantic_memories", user_id)


def episodic_namespace(user_id: str) -> tuple[str, ...]:
    return ("episodic_memories", user_id)


PROCEDURAL_NAMESPACE = ("procedural_memory", "global")
PROCEDURAL_KEY = "playbook"

DEFAULT_PLAYBOOK = (
    "- Always verify identity before discussing billing or account details.\n"
    "- Never approve refunds over $500 without escalating to a human agent.\n"
    "- Be concise. Support chat users want answers, not essays."
)


# --------------------------------------------------------------------------
# Checkpointer (working memory) and Store (Long-term memory) factories
# --------------------------------------------------------------------------

def build_checkpointer(settings: Settings) -> BaseCheckpointSaver:
    if settings.persistence_backend == "postgres":
        # Production path. Requires: pip install langgraph-checkpoint-postgres psycopg[binary]
        from langgraph.checkpoint.postgres import PostgresSaver

        checkpointer_cm = PostgresSaver.from_conn_string(settings.postgres_dsn)
        checkpointer = checkpointer_cm.__enter__()
        checkpointer.setup() # creates tables on first run
        return checkpointer

    # Dev default - wiped on every restart. Fine for local iteration, never for prod.
    return InMemorySaver()


def build_store(settings: Settings) -> BaseStore:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        api_key=SecretStr(settings.google_api_key),
        output_dimensionality=settings.embedding_dims,
    )

    index_config: IndexConfig  = {
        "dims": settings.embedding_dims,
        "embed": embeddings
    }

    if settings.persistence_backend == "postgres":
        from langgraph.store.postgres import PostgresStore
        from langgraph.store.postgres.base import PostgresIndexConfig

        store_cm = PostgresStore.from_conn_string(settings.postgres_dsn, index=cast(PostgresIndexConfig, index_config))
        store = store_cm.__enter__()
        store.setup()
        return store

    return InMemoryStore(index=index_config)


# --------------------------------------------------------------------------
# Semantic memory tools - LangMem's generic hot-path tools.
# The LLM decides to call these; namespace is templated per-user via
# LangGraph's runtime config injection ("{user_id}" resolved at call time).
# --------------------------------------------------------------------------


def semantic_memory_tools():
    return [
        create_manage_memory_tool(namespace=("semantic_memories", "{user_id}")),
        create_search_memory_tool(namespace=("semantic_memories", "{user_id}"))
    ]


class SupportEpisode(BaseModel):
    situation: str = Field(description="What the user's problem or request was")
    action_taken: str = Field(description="What the agent did to resolve it")
    outcome: str = Field(description="How it was resolved and whether the user was satisfied")


@tool
def record_episode(
    episode: SupportEpisode,
    user_id: str,
    store: Annotated[BaseStore, InjectedStore]
) -> str:
    """Record a completed support interaction as an episodic memory.
    Call this once an issue has reached a resolution (fixed, escalated, or
    explicitly dropped by the user) - not for every message."""
    key = str(uuid.uuid4())
    store.put(episodic_namespace(user_id), key, episode.model_dump())
    return f"Episode recorded ({key})."


@tool
def recall_similar_episodes(
    query: str,
    user_id: str,
    store: Annotated[BaseStore, InjectedStore]
) -> str:
    """Search past support episodes for situations similar to the current
    one, to check how they were resolved before responding."""
    results = store.search(episodic_namespace(user_id), query=query, limit=3)
    if not results:
        return "No similar past episodes found."
    return "\n\n".join(
        f"- Situation: {r.value['situation']}\n"
        f"  Action taken: {r.value['action_taken']}\n"
        f"  Outcome: {r.value['outcome']}"
        for r in results
    )


def episodic_memory_tools():
    return [record_episode, recall_similar_episodes]


# --------------------------------------------------------------------------
# Procedural memory - read at prompt-construction time, written by an
# offline reflection job (see app/agents/reflection.py), never by the
# in-conversation agent itself. Treat it as read-only from the hot path.
# --------------------------------------------------------------------------


def get_playbook(store: BaseStore) -> str:
    record = store.get(PROCEDURAL_NAMESPACE, PROCEDURAL_KEY)
    return record.value["rules"] if record else DEFAULT_PLAYBOOK


def ensure_playbook_seeded(store: BaseStore) -> None:
    if store.get(PROCEDURAL_NAMESPACE, PROCEDURAL_KEY) is None:
        store.put(PROCEDURAL_NAMESPACE, PROCEDURAL_KEY, {"rules": DEFAULT_PLAYBOOK})